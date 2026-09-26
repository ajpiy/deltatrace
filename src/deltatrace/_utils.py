"""Shared helpers: key/basis resolution, the cost matrix, CellGraph and the cell graph builders."""

from __future__ import annotations

import functools
import os
import warnings
from collections.abc import Mapping
from types import SimpleNamespace

import anndata as ad
import networkx as nx
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Private helpers (shared with deltatrace._plotting)
# ---------------------------------------------------------------------------
def _resolve_graph(adata, graph=None):
    """Resolve which stored graph (``neighbors_<graph>``) to use, auto-detecting if
    `graph` is None."""
    if graph is not None:
        if f"neighbors_{graph}" not in adata.uns and f"neighbors_{graph}_distances" not in adata.obsp:
            stored = _stored_graphs(adata)
            raise ValueError(
                f"No graph {graph!r} in this object (neighbors_{graph}); it holds "
                f"{stored if stored else 'none'}. 'intr' is built by "
                f"dt.pp.graphs(adata, adata_dist), 'expr' also by dt.pp.graphs(adata)."
            )
        return graph
    candidates = [
        k.removeprefix("neighbors_").removesuffix("_distances")
        for k in adata.obsp
        if k.startswith("neighbors_")
        and k.endswith("_distances")
        and k != "neighbors_distances"
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "No DeltaTrace graph found. Run dt.pp.graphs(adata, adata_dist) first, or "
            "dt.pp.graphs(adata) for the expression graph alone (no spatial input)."
        )
    raise ValueError(
        f"Multiple integrated graphs found: {candidates}. Pass graph= explicitly."
    )


def _stored_graphs(adata):
    """Names of the graphs stored as ``neighbors_<graph>`` (scanpy's own ``neighbors`` aside)."""
    return [k.removeprefix("neighbors_").removesuffix("_distances") for k in adata.obsp
            if k.startswith("neighbors_") and k.endswith("_distances")
            and k != "neighbors_distances"]


def _as_labels(clusters, name):
    """Cluster labels as a list of str (obs labels are compared as strings), accepting a
    single label given as a bare string."""
    if isinstance(clusters, str):
        return [clusters]
    try:
        return [str(c) for c in clusters]
    except TypeError:
        raise TypeError(f"{name} must be a list of cluster labels, got {clusters!r}") from None


def _check_labels(adata, cluster_key, **roles):
    """Fail early and clearly when obs[cluster_key] is missing or a role's labels are not in
    it, rather than later as a bare KeyError or an empty-pool error."""
    if cluster_key not in adata.obs:
        raise KeyError(
            f"No column {cluster_key!r} in adata.obs to read the cluster labels from. Pass "
            f"cluster_key=<column>; obs has {list(adata.obs.columns)[:12]}."
        )
    present = set(adata.obs[cluster_key].astype(str))
    for role, clusters in roles.items():
        missing = [c for c in (clusters or []) if c not in present]
        if missing:
            shown = sorted(present)[:25]
            raise ValueError(
                f"{role} {missing} not found in obs[{cluster_key!r}], whose labels are "
                f"{shown}{' ...' if len(present) > len(shown) else ''}."
            )


def _resolve_basis(adata, basis):
    """Return the (n_obs, n_dims) embedding coordinates for `basis` from adata.obsm."""
    key = f"X_{basis}"
    if key in adata.obsm:
        return adata.obsm[key]
    if basis in adata.obsm:
        return adata.obsm[basis]
    raise KeyError(
        f"Coordinates for '{basis}' not found in obsm. "
        f"Expected 'X_{basis}' or '{basis}'. Compute it first."
    )


def _off_lcc_mask(adata, graph=None):
    """Boolean mask (n_obs,), True for cells outside the largest connected component
    of neighbors_<graph>. None if no such graph is stored."""
    import scipy.sparse.csgraph as csg

    try:
        graph = _resolve_graph(adata, graph)
        dist_key = adata.uns[f"neighbors_{graph}"]["distances_key"]
        D = adata.obsp[dist_key].tocsr()
    except (ValueError, KeyError):
        return None

    D = (D + D.T) / 2
    if D.diagonal().any():           # setdiag on an empty diagonal inserts zeros (scipy<1.13 warns)
        D.setdiag(0)
    D.eliminate_zeros()
    _, labels = csg.connected_components(D, directed=False)
    if labels.size == 0:
        return None
    return labels != np.bincount(labels).argmax()


def _check_costs(D, dist_key):
    """Refuse a cost matrix with non-finite or negative entries.

    Costs are ``clip(-ln(connectivity), floor, cap)``, so ``pp.integrate`` can only
    produce finite, positive values; a NaN or negative entry means the stored
    ``obsp[dist_key]`` was built or edited by hand. Downstream it would not fail loudly:
    scipy's Dijkstra returns arbitrary endpoints on NaN and never terminates on a
    negative edge (it allocates without bound), and the transport solve would proceed.
    """
    data = D.data
    n_nonfinite = int((~np.isfinite(data)).sum())
    n_negative = int((data < 0).sum())            # NaN < 0 is False, so no double count
    if n_nonfinite or n_negative:
        raise ValueError(
            f"obsp['{dist_key}'] holds {n_nonfinite} non-finite and {n_negative} negative "
            "edge cost(s); every cost must be finite and >= 0. pp.integrate() never "
            "writes such values, so this matrix was built or modified by hand -- fix it "
            "(drop the edge, or set a positive cost) before running endpoints/qrot."
        )


def _cost_matrix(adata, graph=None):
    """Symmetrized (n_obs, n_obs) cell-cell transport cost matrix underlying the QROT
    graph; scipy.sparse.csgraph form of what _build_digraph exposes as edge "cost"."""
    graph = _resolve_graph(adata, graph)
    dist_key = adata.uns[f"neighbors_{graph}"]["distances_key"]
    D = adata.obsp[dist_key].tocsr()
    _check_costs(D, dist_key)
    D = (D + D.T) / 2
    if D.diagonal().any():           # setdiag on an empty diagonal inserts zeros (scipy<1.13 warns)
        D.setdiag(0)
    D.eliminate_zeros()
    return D


def _build_digraph(adata: ad.AnnData, graph: str | None = None) -> nx.DiGraph:
    """Build a bidirectional cell-cell nx.DiGraph from a stored KNN graph, with
    "cost" (symmetrized distance) and "weight" (symmetrized connectivity) edge
    attributes, and obs_names/name_to_idx/idx_to_name on G.graph."""
    graph = _resolve_graph(adata, graph)
    neighbors_key = f"neighbors_{graph}"
    conn_key = adata.uns[neighbors_key]["connectivities_key"]
    dist_key = adata.uns[neighbors_key]["distances_key"]

    A = adata.obsp[conn_key].tocsr()
    D = adata.obsp[dist_key].tocsr()
    _check_costs(D, dist_key)

    A = (A + A.T) / 2
    D = (D + D.T) / 2
    if A.diagonal().any():           # setdiag on an empty diagonal inserts zeros (scipy<1.13 warns)
        A.setdiag(0)
    A.eliminate_zeros()
    if D.diagonal().any():           # setdiag on an empty diagonal inserts zeros (scipy<1.13 warns)
        D.setdiag(0)
    D.eliminate_zeros()

    obs_names = adata.obs_names.to_numpy()
    name_to_idx = {str(nm): i for i, nm in enumerate(obs_names)}

    G = nx.from_scipy_sparse_array(D, create_using=nx.DiGraph, edge_attribute="cost")

    A_coo = A.tocoo()
    nx.set_edge_attributes(
        G,
        {
            (int(u), int(v)): float(w)
            for u, v, w in zip(A_coo.row, A_coo.col, A_coo.data, strict=True)
            if G.has_edge(int(u), int(v))
        },
        name="weight",
    )

    G.graph["obs_names"] = obs_names
    G.graph["name_to_idx"] = name_to_idx
    G.graph["idx_to_name"] = {i: str(nm) for i, nm in enumerate(obs_names)}

    return G


class CellGraph:
    """The bidirectional cell-cell graph :func:`endpoints` samples on and :func:`qrot`
    solves over, held as scipy/numpy arrays instead of a networkx.DiGraph.

    Attributes
    ----------
    cost : scipy.sparse CSR matrix, (n_obs, n_obs)
        The symmetrized, zero-diagonal transport cost matrix, built by
        :func:`_cost_matrix`. Every geodesic in :func:`endpoints` runs on it.
    edge_array : ndarray of int64, (m, 2)
        Directed edges ``(u, v)``, one per stored entry of ``cost``, in CSR storage
        order. That is the order networkx inserted them in, so this is the old
        ``list(G.edges())``, and the order qrot's flow ``J`` is indexed by. It is not
        called ``edges`` on purpose: networkx code such as ``(u, v) in G.edges`` would
        run on an array and quietly answer something else.
    cost_vec : ndarray of float64, (m,)
        ``cost``'s stored values aligned with ``edge_array`` (the old edge ``"cost"``).
    graph : dict
        What the old ``G.graph`` held: ``obs_names``, ``name_to_idx``, ``idx_to_name``,
        plus what :func:`endpoints` adds (source/sink indices and names,
        ``disconnected_info``).
    n, m : int
        Number of cells (nodes 0..n-1) and of directed edges.

    ``to_networkx()`` returns the networkx.DiGraph ``endpoints`` used to return. Any other
    networkx attribute (``G.nodes``, ``G.edges``, ...) raises an AttributeError saying so.
    """

    __slots__ = ("_nx_args", "cost", "cost_vec", "edge_array", "graph")

    def __init__(self, cost, edge_array, cost_vec, graph, nx_args):
        self.cost = cost
        self.edge_array = edge_array
        self.cost_vec = cost_vec
        self.graph = graph
        self._nx_args = nx_args

    def __getattr__(self, name):
        # Only reached for a name CellGraph lacks. It must not read self: copy and pickle
        # look names up here before any slot is set.
        hint = (" endpoints()['G'] is a CellGraph now, not a networkx.DiGraph: call "
                ".to_networkx() for that graph." if hasattr(nx.DiGraph, name) else "")
        raise AttributeError(f"'CellGraph' object has no attribute {name!r}.{hint}")

    @property
    def graph_name(self) -> str:
        """The stored graph (``neighbors_<graph_name>``) this cell graph was built from."""
        return self._nx_args[1]

    @property
    def n(self) -> int:
        return int(self.cost.shape[0])

    @property
    def m(self) -> int:
        return int(self.edge_array.shape[0])

    def __repr__(self) -> str:
        return f"CellGraph(n={self.n}, m={self.m} directed edges)"

    def to_networkx(self) -> nx.DiGraph:
        """The networkx.DiGraph :func:`endpoints` used to return: rebuilt by the unchanged
        :func:`_build_digraph` from copies of the two stored matrices taken when this graph
        was built (so an in-place edit of ``adata.obsp`` since then does not reach it, as it
        could not reach the DiGraph), then given this graph's ``.graph`` metadata. Same
        nodes, edge order, "cost" and "weight" edge attributes (edges the connectivities do
        not store carry no "weight")."""
        G = _build_digraph(*self._nx_args)
        G.graph.update(self.graph)
        return G


def _build_cell_graph(adata: ad.AnnData, graph: str | None = None) -> CellGraph:
    """Build the :class:`CellGraph` :func:`endpoints` works on: the graph
    :func:`_build_digraph` builds, from the same cost matrix, with no networkx."""
    graph = _resolve_graph(adata, graph)
    neighbors_key = f"neighbors_{graph}"
    # Read what _build_digraph reads, in its order, so a missing key or a dense matrix
    # fails the same way. The connectivities are only kept for to_networkx().
    conn_key = adata.uns[neighbors_key]["connectivities_key"]
    dist_key = adata.uns[neighbors_key]["distances_key"]
    A = adata.obsp[conn_key].tocsr()

    D = _cost_matrix(adata, graph)
    if type(D) not in (sp.csr_matrix, sp.csr_array):
        # From an AnnData view, D keeps anndata's view class, whose deepcopy raises (so
        # copy.deepcopy of the graph result would). anndata's copy() of it is the plain
        # scipy matrix: same values, same storage order.
        D = D.copy()
    nnz = D.nnz
    # One edge per stored entry, in storage order: the order nx.from_scipy_sparse_array
    # fed add_edges_from. Copies, so nothing done to D later can move them.
    rows = np.repeat(np.arange(D.shape[0], dtype=np.int64), np.diff(D.indptr))
    edges = np.column_stack([rows, D.indices[:nnz].astype(np.int64)])
    # same_kind: a complex matrix raises instead of silently dropping its imaginary part
    cost_vec = D.data[:nnz].astype(np.float64, casting="same_kind")

    obs_names = adata.obs_names.to_numpy()
    meta = {
        "obs_names": obs_names,
        "name_to_idx": {str(nm): i for i, nm in enumerate(obs_names)},
        "idx_to_name": {i: str(nm) for i, nm in enumerate(obs_names)},
    }
    # What _build_digraph will need to rebuild the same DiGraph on demand: the two stored
    # matrices and the names, not the whole AnnData. The matrices are copied now, so an
    # in-place edit of adata.obsp after endpoints() cannot reach to_networkx(), as it could
    # not reach the DiGraph built here before. (.copy(), not deepcopy: anndata deep-copies
    # a view's matrix into its parent's whole obsp.)
    source = SimpleNamespace(
        uns={neighbors_key: {"connectivities_key": conn_key, "distances_key": dist_key}},
        obsp={conn_key: A.copy(), dist_key: adata.obsp[dist_key].copy()},
        obs_names=adata.obs_names,
    )
    return CellGraph(D, edges, cost_vec, meta, (source, graph))


# Graph mode -> the graph name it is stored under (neighbors_<graph>). Single source of
# truth for integrate() and graphs().
_MODE_GRAPHS = {
    "intersection": "intr",
    "expression": "expr",
}


# Warnings raised inside the package skip its own frames, so they point at the caller's line
# however many wrappers (decorators, tl.trajectory) sit in between.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _graph_from_key(key) -> str:
    """The graph a pre-0.1 ``reconnect_key`` named: ``"neighbors_expr"`` -> ``"expr"``. Any
    other key (a scanpy graph stored with ``key_added=<key>``) passes through unchanged, and
    :func:`_graph_distances_key` finds it under either spelling."""
    if not isinstance(key, str) or not key.removeprefix("neighbors_"):
        raise ValueError(
            f"reconnect_key={key!r} does not name a graph. Pass reconnect_graph=<graph> "
            f"instead, e.g. reconnect_graph='expr'."
        )
    return key.removeprefix("neighbors_")


def _graph_distances_key(adata, graph: str) -> str | None:
    """``obsp`` key of the distances of the graph named ``graph``: a DeltaTrace graph
    (``neighbors_<graph>``) first, else a scanpy one stored with ``key_added=<graph>``
    (``uns[<graph>]["distances_key"]``, or ``<graph>_distances``). None if neither exists."""
    for uns_key in (f"neighbors_{graph}", graph):
        entry = adata.uns.get(uns_key)
        key = entry.get("distances_key") if isinstance(entry, Mapping) else None
        for cand in (key, f"{uns_key}_distances"):
            if cand is not None and cand in adata.obsp:
                return cand
    return None


def _renamed_kwargs(**renames):
    """Keep accepting keyword names used before 0.1.0, with a FutureWarning.

    Each value is the new parameter name, or ``(new_name, convert)`` where ``convert``
    maps the old value onto the new parameter's (``reconnect_key="neighbors_expr"`` ->
    ``reconnect_graph="expr"``). Passing both names is a TypeError. Apply it outside
    ``_log.stage`` so the stage binds the new names.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for old, spec in renames.items():
                if old not in kwargs:
                    continue
                new, convert = spec if isinstance(spec, tuple) else (spec, None)
                if new in kwargs:
                    raise TypeError(
                        f"{fn.__name__}() got both {old}= and {new}=; pass only {new}=."
                    )
                value = kwargs.pop(old)
                kwargs[new] = convert(value) if convert is not None else value
                warnings.warn(
                    f"{fn.__name__}(): {old}= is deprecated and will be removed; "
                    f"use {new}={kwargs[new]!r} instead.",
                    FutureWarning,
                    stacklevel=2,
                )
            return fn(*args, **kwargs)

        return wrapper

    return deco


def _suffix(key_added: str | None) -> str:
    return "" if key_added is None else f"_{key_added}"


def _is_positional_index(names) -> bool:
    """True if `names` is a default 0, 1, 2, ... index (no cell identity)."""
    return all(str(nm) == str(i) for i, nm in enumerate(names))
