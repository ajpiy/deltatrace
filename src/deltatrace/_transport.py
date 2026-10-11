"""tl: the transport solve (qrot), the velocity grid, and trajectory() composing the stages."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spla
from scipy.optimize import linprog, minimize

from . import _log
from ._endpoints import endpoints
from ._lineages import _check_terminal_cut, lineages, suggest_omega, terminality
from ._utils import (
    _PKG_DIR,
    CellGraph,
    _as_labels,
    _check_labels,
    _graph_distances_key,
    _graph_from_key,
    _off_lcc_mask,
    _renamed_kwargs,
    _resolve_basis,
    _resolve_graph,
    _suffix,
)


# ---------------------------------------------------------------------------
# tl.qrot  (+ pseudotime reconnection helper)
# ---------------------------------------------------------------------------
def _reconnect_pseudotime(
    adata_basis,
    adata_clust,
    pt_dict: dict,
    graph: str = "intr",
    reconnect_graph: str = "expr",
) -> dict:
    """Give every off-LCC cell the pseudotime of its nearest connected cell, by
    Dijkstra on the reconnect_graph graph (unreachable cells fall back to the LCC
    median). Returns the full {cell name: pseudotime} mapping."""
    disc_mask = _off_lcc_mask(adata_basis, graph)

    if disc_mask is None:
        disc_col = f"disconnected_{graph}"
        fallback_col = "disconnected_intr"
        if disc_col in adata_basis.obs.columns:
            disc_mask = adata_basis.obs[disc_col].astype(bool).values
        elif fallback_col in adata_basis.obs.columns:
            disc_mask = adata_basis.obs[fallback_col].astype(bool).values
        else:
            _log.detail("No disconnected nodes: all pseudotimes already assigned.")
            return dict(pt_dict)

    disc_names = list(adata_basis.obs_names[disc_mask])

    if len(disc_names) == 0:
        _log.detail("No off-LCC nodes: all pseudotimes already assigned.")
        return dict(pt_dict)

    clust_names = adata_clust.obs_names
    name_to_idx = {name: i for i, name in enumerate(clust_names)}

    donor = ~clust_names.isin(disc_names) & clust_names.isin(list(pt_dict))
    connected_names_list = list(clust_names[donor])
    connected_indices = np.flatnonzero(donor)

    dist_key = _graph_distances_key(adata_clust, reconnect_graph)
    if dist_key is None:
        raise ValueError(
            f"No graph {reconnect_graph!r} to reconnect off-LCC cells through. Pass a graph "
            "from pp.graphs (e.g. reconnect_graph='expr') or a scanpy key_added."
        )
    D_clust = adata_clust.obsp[dist_key].tocsr()

    dist_sym = (D_clust + D_clust.T) / 2
    if dist_sym.diagonal().any():    # setdiag on an empty diagonal inserts zeros (scipy<1.13 warns)
        dist_sym.setdiag(0)
    dist_sym.eliminate_zeros()

    _targets = [n for n in sorted(disc_names) if n in name_to_idx]
    if _targets and not connected_indices.size:
        raise ValueError(
            "No LCC cell of adata_clust has a pseudotime: no donor to reconnect the "
            f"{len(_targets)} off-LCC cells from."
        )
    _log.detail(f"Reconnecting {len(_targets)} off-LCC cells from "
                f"{len(connected_indices)} LCC donors...")
    # Each target's distance to its nearest donor, then one Dijkstra from the target cut
    # just past it: argmin over the donors in obs order resolves a tie to the first donor.
    # The 1e-9 slack only absorbs the two directions summing a path in a different order.
    d_nearest = csg.dijkstra(dist_sym, indices=connected_indices, min_only=True)

    pseudotime_full = dict(pt_dict)
    assigned = 0
    fallback = []

    for disc_name in _targets:
        t = name_to_idx[disc_name]
        if np.isfinite(d_nearest[t]):
            limit = d_nearest[t] * (1 + 1e-9) + 1e-300
            dists = csg.dijkstra(dist_sym, indices=t, limit=limit)[connected_indices]
            pseudotime_full[disc_name] = pseudotime_full[
                connected_names_list[int(np.argmin(dists))]
            ]
            assigned += 1
        else:
            fallback.append(disc_name)

    if fallback:
        connected_pts = [pseudotime_full[n] for n in connected_names_list]
        median_pt = float(np.median(connected_pts))
        for disc_name in fallback:
            pseudotime_full[disc_name] = median_pt
        warnings.warn(
            f"{len(fallback)} cell(s) outside the '{graph}' graph's largest connected "
            f"component are unreachable through reconnect_graph={reconnect_graph!r} too; "
            f"they get the median pseudotime ({median_pt:.3g}) as a placeholder.",
            UserWarning,
            skip_file_prefixes=(_PKG_DIR,),
        )

    if _log.wants_detail():
        n_total = len(adata_basis.obs_names)
        _log.detail(
            f"pseudotime reassignment: {len(connected_names_list)} connected, "
            f"{assigned} reassigned via geodesic"
            + (f", {len(fallback)} median fallback" if fallback else "")
            + f" ({len(pseudotime_full)}/{n_total} total)"
        )

    return pseudotime_full


def _median_off_lcc_pseudotime(adata_basis, pt_dict: dict, graph: str | None = None) -> dict:
    """Set every off-LCC cell's pseudotime to the median of the LCC cells' (an
    explicit, reproducible placeholder, not an inferred position). No-op if
    adata_basis is None or there are no off-LCC cells."""
    pt_full = dict(pt_dict)
    if adata_basis is None:
        return pt_full

    off_mask = _off_lcc_mask(adata_basis, graph)
    if off_mask is None:
        _log.detail("No graph to locate the LCC: off-LCC pseudotimes left as solved.")
        return pt_full

    names = np.asarray([str(nm) for nm in adata_basis.obs_names])
    off_names = [nm for nm in names[off_mask] if nm in pt_full]
    if not off_names:
        _log.detail("No off-LCC cells: every pseudotime came from the solved LCC.")
        return pt_full

    lcc_vals = np.array([pt_full[nm] for nm in names[~off_mask] if nm in pt_full], float)
    lcc_vals = lcc_vals[np.isfinite(lcc_vals)]
    if lcc_vals.size == 0:
        _log.detail(f"No finite LCC pseudotime: {len(off_names)} off-LCC cells left as is.")
        return pt_full

    median_pt = float(np.median(lcc_vals))
    for nm in off_names:
        pt_full[nm] = median_pt
    _log.detail(f"reconnect=False: set {len(off_names)} off-LCC cells to the LCC median "
                f"pseudotime {median_pt:.6f} (from {lcc_vals.size} LCC cells)")
    return pt_full


# ---------------------------------------------------------------------------
# QROT solver: the dual, by SciPy (trust region, Newton, midpoint of the free potentials)
# ---------------------------------------------------------------------------
def _solve_qrot(tail, head, c, f, tau, gtol=1e-4, tol=1e-12, eps=1e-8, max_iter=300):
    """Optimal flow J and potentials p of the QROT dual (see qrot)."""
    if not (np.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be a positive finite number, got {tau!r}.")
    n, m = len(f), len(tail)
    scale = np.abs(f).max()

    # demand must balance on every graph piece
    _, lab = csg.connected_components(sp.coo_matrix((np.ones(m), (tail, head)), shape=(n, n)), directed=False)
    if scale == 0 or np.any(np.abs(np.bincount(lab, f)) > tol * scale * np.bincount(lab)):
        raise ValueError("The demand (sinks minus sources) is zero or does not balance on a connected piece "
                         "of the graph.")

    def flow(p):
        return np.maximum(p[head] - p[tail] - c, 0.0) / tau

    def residual(J):
        return f - np.bincount(head, J, minlength=n) + np.bincount(tail, J, minlength=n)

    def gain(p, d, t, r):
        # dual(p + t d) - dual(p), summed per edge to avoid cancellation
        s0 = p[head] - p[tail] - c
        ds = t * (d[head] - d[tail])
        s1 = s0 + ds
        on0, on1 = s0 > 0, s1 > 0
        term = np.where(on0 & on1, -ds**2, 0.0)
        term = np.where(on0 & ~on1, s0 * (2 * ds + s0), term)
        term = np.where(~on0 & on1, -s1**2, term)
        return t * (r @ d) + term.sum() / (2 * tau)

    # tau continuation near the LP limit (tau max|f| < max c / 10)
    k = max(np.ceil(np.log10(0.1 * c.max() / (tau * scale))), 0) if c.max() > 0 else 0
    taus = tau * 10.0 ** np.arange(k, -1, -1)
    tau = taus[0]

    # start: shortest-path distance from the sources
    G = sp.csr_matrix((c, (tail, head)), shape=(n, n))
    dist = csg.dijkstra(G, indices=np.flatnonzero(f < 0), min_only=True)
    reached = np.isfinite(dist)
    p = np.where(reached, dist, dist[reached].max())

    # trust region (state cached per point); its model uses Newton's edges, s > -eps, so the shortest-path
    # edges at their cost give it curvature from the start
    cache = {}

    def state(p):
        if not np.array_equal(cache.get("p"), p):
            s = p[head] - p[tail] - c
            a = s > -eps
            cache.update(p=p.copy(), J=np.maximum(s, 0.0) / tau, tail=tail[a], head=head[a])
        return cache

    def neg_dual(p):
        J = state(p)["J"]
        return -(f @ p - 0.5 * tau * (J @ J)) / scale

    def hessp(p, v):
        st = state(p)
        dv = (v[st["head"]] - v[st["tail"]]) / tau
        return (np.bincount(st["head"], dv, minlength=n) - np.bincount(st["tail"], dv, minlength=n)) / scale

    tr = minimize(neg_dual, p, jac=lambda p: -residual(state(p)["J"]) / scale,
                  hessp=hessp, method="trust-krylov", options=dict(gtol=gtol, maxiter=1000))
    p = tr.x

    # newton refinement
    for tau in taus:
        for _ in range(max_iter):
            s = p[head] - p[tail] - c
            r = residual(np.maximum(s, 0.0) / tau)
            # converged: residual below tol or within rounding
            rnd = np.finfo(float).eps * (np.abs(p[head]) + np.abs(p[tail]))
            w = np.where(s > -rnd, rnd / tau, 0.0)
            if np.all(np.abs(r) <= tol * scale + np.bincount(head, w, minlength=n) + np.bincount(tail, w, minlength=n)):
                break

            # hessian: laplacian of the active edges
            a = s > -eps
            A = sp.coo_matrix((np.full(a.sum(), 1.0 / tau), (tail[a], head[a])), shape=(n, n))
            L = csg.laplacian((A + A.T).tocsr()).tocsr()

            # newton step per piece, grounded at one cell
            k, lab = csg.connected_components(L, directed=False)
            size = np.bincount(lab)
            mean = np.bincount(lab, r) / size
            mean[np.abs(mean) <= tol * scale] = 0.0
            idx = np.flatnonzero(L.diagonal() > 0)
            _, first = np.unique(lab[idx], return_index=True)
            free = np.setdiff1d(idx, idx[first])
            d = np.zeros(n)
            d[free] = spla.splu(L[free][:, free].tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                options=dict(SymmetricMode=True)).solve((r - mean[lab])[free])

            # shift pieces with a net residual until a boundary edge opens
            if mean.any():
                cut = lab[tail] != lab[head]
                gap_in, gap_out = np.full(k, np.inf), np.full(k, np.inf)
                np.minimum.at(gap_in, lab[head[cut]], -s[cut])
                np.minimum.at(gap_out, lab[tail[cut]], -s[cut])
                move = np.where(mean > 0, gap_in, np.where(mean < 0, -gap_out, 0.0)) + tau * mean * size
                move = np.where(np.isfinite(move), move, 0.0) - np.where(mean != 0, np.bincount(lab, d) / size, 0.0)
                d += move[lab]

            # armijo line search
            t, slope = 1.0, r @ d
            while gain(p, d, t, r) < 1e-4 * t * slope and t > 1e-12:
                t /= 2
            p = p + t * d
        else:
            B = sp.coo_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[tail, head], np.r_[np.arange(m), np.arange(m)])), shape=(n, m))
            if linprog(np.zeros(m), A_eq=B, b_eq=f / scale, method="highs").status == 2:
                raise ValueError("The demand is infeasible: along the edge directions, the sources' mass cannot reach the sinks.")
            raise RuntimeError(f"QROT did not converge: Kirchhoff residual {np.abs(r).max():.2e} after {max_iter} Newton steps")

    # zero unresolvable flows, then center the free potentials
    J = flow(p)
    J[J <= tol * scale + np.finfo(float).eps * (np.abs(p[head]) + np.abs(p[tail])) / tau] = 0.0
    off = np.abs(residual(J)).max()
    _log.detail(f"solver: dual trust region ({tr.nit} steps) + Newton, Kirchhoff residual {off:.1e}")
    if off > 1e-6 * scale:
        warnings.warn(f"The flow misses the demand by {off / scale:.0e} of its size, below the solver's "
                      "resolution: raise tau, or scale the costs and masses toward 1.",
                      RuntimeWarning, skip_file_prefixes=(_PKG_DIR,))
    return J, _midpoint_free_potentials(tail, head, c, J, p, tau)


def _midpoint_free_potentials(tail, head, c, J, p, tau):
    """Potentials the flow leaves free, at the midpoint of their optimal range; NaN where unbounded."""
    n, on = len(p), J > 0
    if not on.any():
        return np.full(n, np.nan)

    # slack of every edge at p (>= 0), plus zero-slack reverse edges along the flow; parallel edges keep the smallest
    t, h = np.r_[tail, head[on]], np.r_[head, tail[on]]
    s = np.r_[np.maximum(c + tau * J - (p[head] - p[tail]), 0.0), np.zeros(on.sum())]
    o = np.lexsort((s, h, t))
    o = o[np.r_[True, (np.diff(t[o]) != 0) | (np.diff(h[o]) != 0)]]
    G = sp.csr_matrix((s[o], (t[o], h[o])), shape=(n, n))

    # reference: one flow group per graph piece
    cells = np.unique(np.r_[tail[on], head[on]])
    _, group = csg.connected_components(sp.coo_matrix((np.ones(on.sum()), (tail[on], head[on])), shape=(n, n)), directed=False)
    _, piece = csg.connected_components(sp.coo_matrix((np.ones(len(tail)), (tail, head)), shape=(n, n)), directed=False)
    ref = cells[np.isin(group[cells], group[cells[np.unique(piece[cells], return_index=True)[1]]])]

    # latest and earliest optimal potential of every cell, then their midpoint
    up = csg.dijkstra(G, indices=ref, min_only=True)
    down = csg.dijkstra(G.T.tocsr(), indices=ref, min_only=True)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(up) & np.isfinite(down), p + (up - down) / 2, np.nan)


def _normalize_costs(c, cost_range):
    """Min-max the edge costs into cost_range; a constant-cost graph maps to cost_range[0]."""
    lo, hi = cost_range
    if not (np.isfinite([lo, hi]).all() and 0 <= lo <= hi):
        raise ValueError(f"cost_range must satisfy 0 <= low <= high < inf, got {cost_range!r}.")
    span = c.max() - c.min()
    scale = (hi - lo) / (span if span != 0 else 1.0)
    c = c * scale + (lo - c.min() * scale)
    _log.detail(f"Costs normalized to [{c.min():.2e}, {c.max():.2e}]")
    return c


@_renamed_kwargs(tag="graph", reconnect_key=("reconnect_graph", _graph_from_key))
@_log.stage("solving quadratically regularized transport")
def qrot(
    graph_result: dict,
    tau: float = 100,
    normalize_costs: bool = True,
    cost_range: tuple = (1e-6, 1.0),
    verbose: int | bool = 1,
    adata_basis=None,
    adata_clust=None,
    reconnect: bool = True,
    reconnect_graph: str = "expr",
    graph: str | None = None,
    key_added: str | None = None,
) -> None:
    """Transport flow from source to sink cells and read pseudotime off the potentials.

    Quadratically regularized optimal transport (QROT) on the directed cell graph. The
    primal pushes a flow J along the edges at cost c, meeting the demand f that
    :func:`endpoints` built:

        min_{J >= 0}   <c, J> + (tau/2) ||J||^2
        s.t.           B J = f

    B is the cell-edge incidence matrix (for an edge e = u -> v, B[u, e] = -1 and
    B[v, e] = +1). The dual has one potential p per cell:

        max_p   <f, p> - (1 / (2 tau)) || (B^T p - c)_+ ||^2

    tied to the primal by the slackness relation ``J = (B^T p - c)_+ / tau``, with
    ``(B^T p)_e = p_v - p_u``: an edge carries flow only where ``p_v > p_u + c_e``, so
    the potential rises strictly from sources to sinks. Min-max scaled to ``[0, 1]``,
    p IS the pseudotime.

    The dual is solved directly (SciPy trust region, then Newton steps on the Laplacian of
    the flow edges) to a Kirchhoff residual of 1e-12 relative to the demand (or its rounding
    floor). Cells no flow touches, and the offsets between groups of cells the flow joins,
    are only bounded by ``p_v - p_u <= c_e``; each is placed at the midpoint of its earliest
    and latest optimal value (two shortest-path searches on the edge slacks).

    Parameters
    ----------
    graph_result : dict
        Output of :func:`endpoints` (keys ``"G"``, ``"cost_vec"``, ``"f"``), or of
        ``CellGraph.to_networkx()``, whose result is a networkx.DiGraph.
    tau : float, default 100
        Regularization strength; larger ``tau`` opens more edges and spreads the flow.
    normalize_costs, cost_range
        Min-max scale edge costs into ``cost_range`` before solving.
    verbose : int or bool, default 1
        Logging verbosity.
    adata_basis : anndata.AnnData, optional
        If given, writes ``obs["pseudotime_raw"]`` (before reconnection; NaN for cells
        in graph pieces without a source or sink, which have no potential, and, with
        one-way edges, for cells bounded on one side only),
        ``obs["pseudotime"]`` (after) and ``uns["dt_flow"]`` (``{"J": (n_edges,),
        "edges": (n_edges, 2)}``).
    adata_clust : anndata.AnnData, optional
        Required when ``reconnect=True``, for the geodesic reconnection graph.
    reconnect : bool, default True
        Give off-LCC cells a pseudotime via nearest-LCC reconnection
        (:func:`_reconnect_pseudotime`).

        WARNING: with ``reconnect=False`` those cells are instead set to the
        median of the LCC cells' pseudotime -- an explicit placeholder, NOT an
        inferred position. Identify them via ``endpoints``' ``disconnected_info``
        before reading anything into it.
    reconnect_graph : str, default "expr"
        Graph in ``adata_clust`` whose geodesics reconnect off-LCC cells: a DeltaTrace
        graph (``neighbors_<reconnect_graph>``), or else a scanpy one stored with
        ``sc.pp.neighbors(key_added=reconnect_graph)``.
    graph : str, optional
        The stored graph ``endpoints`` sampled on (``neighbors_<graph>``), used to find
        the off-LCC cells; auto-detected from ``adata_basis`` if ``None``.
    key_added : str, optional
        Suffix every written key with ``_<key_added>``, so two runs can coexist in
        one object. ``None`` writes the flat keys.

    Returns
    -------
    None
        Results are written into ``adata_basis`` (when provided).
    """
    _sfx = _suffix(key_added)

    G = graph_result["G"]
    c = graph_result["cost_vec"]
    f = graph_result["f"]

    if adata_basis is not None:
        graph = _resolve_graph(adata_basis, graph)

    if isinstance(G, CellGraph):
        # endpoints' graph: its nodes are cells 0..n-1 in order, so the node map below
        # would be the identity, and G.edge_array is the old list(G.edges()) as an (m, 2)
        # array.
        node_map = None
        n = G.n
        E = G.edge_array
        m = G.m
    else:  # a networkx.DiGraph (CellGraph.to_networkx())
        nodes = list(G.nodes())
        node_map = {v: i for i, v in enumerate(nodes)}
        n = len(nodes)

        edges = list(G.edges())
        m = len(edges)
    c = np.asarray(c, dtype=float).ravel()

    assert len(c) == m, f"cost length {len(c)} != edges {m}"

    if normalize_costs and m > 0:
        c = _normalize_costs(c, cost_range)

    # Edge tails u and heads v (the incidence matrix B of the docstring, kept as index arrays).
    if node_map is None:
        us, vs = E[:, 0], E[:, 1]
    else:
        us = np.array([node_map[u] for u, _ in edges], dtype=np.int64)
        vs = np.array([node_map[v] for _, v in edges], dtype=np.int64)

    f_arr = np.asarray(f, dtype=float)
    if f_arr.shape != (n,):
        raise ValueError(
            f"qrot: demand f must be a 1-D vector with one entry per cell ({n}), got "
            f"shape {f_arr.shape}."
        )
    if node_map is None:
        f_vec = f_arr.copy()
    else:
        f_vec = np.zeros(n, dtype=float)
        for v, i in node_map.items():
            f_vec[i] = f_arr[v]

    J_opt, p = _solve_qrot(us, vs, c, f_vec, tau)

    _log.detail(f"QROT solved: {m} edges, tau={tau}")
    _log.detail(f"  Flow range: [{J_opt.min():.4e}, {J_opt.max():.4e}]")
    _log.detail(f"  Potential range: [{np.nanmin(p):.4e}, {np.nanmax(p):.4e}]")

    idx_to_name = G.graph["idx_to_name"]
    if node_map is None:
        pt_raw = {idx_to_name[i]: p[i] for i in range(n)}
    else:
        pt_raw = {idx_to_name[node]: p[i] for node, i in node_map.items()}

    # Min-max the potentials onto [0, 1] (pseudotime); cells without one stay NaN
    _vals = np.array(list(pt_raw.values()))
    _pmin = np.nanmin(_vals)
    _rng = (np.nanmax(_vals) - _pmin) or 1.0
    pt_norm = {k: (v - _pmin) / _rng for k, v in pt_raw.items()}

    if adata_basis is not None:
        obs_names = adata_basis.obs_names.astype(str)
        adata_basis.obs[f"pseudotime_raw{_sfx}"] = (
            pd.Series(pt_norm).reindex(obs_names).to_numpy(float)
        )

    if reconnect and adata_clust is not None:
        pt_full = _reconnect_pseudotime(
            adata_basis,
            adata_clust,
            pt_norm,
            graph=graph,
            reconnect_graph=reconnect_graph,
        )
    elif not reconnect:
        pt_full = _median_off_lcc_pseudotime(adata_basis, pt_norm, graph=graph)
    else:
        pt_full = dict(pt_norm)

    if adata_basis is not None:
        adata_basis.obs[f"pseudotime{_sfx}"] = (
            pd.Series(pt_full).reindex(obs_names).to_numpy(float)
        )
        if node_map is None:
            # np.array of an empty edge list is shape (0,), not (0, 2): keep that
            edges_arr = E.copy() if m else np.array([], dtype=np.int64)
        else:
            edges_arr = np.array(edges, dtype=np.int64)
        adata_basis.uns[f"dt_flow{_sfx}"] = {
            "J": J_opt,
            "edges": edges_arr,
        }


@_log.stage("solving quadratically regularized transport")
def qrot_graph(
    graph,
    a,
    b,
    tau: float = 100,
    directed: bool = False,
    normalize_costs: bool = True,
    verbose: int | bool = 1,
) -> dict:
    """QROT on any graph in SciPy's csgraph form: flow, potentials and pseudotime.

    The solver behind :func:`qrot`, without AnnData or :func:`endpoints`. Quadratically regularized
    optimal transport (QROT) moves the mass ``a`` to ``b`` along the edges: it minimizes
    ``sum_e c_e J_e + (tau / 2) sum_e J_e ** 2`` over flows ``J >= 0`` whose inflow minus outflow is
    ``b - a`` at every node. Its dual gives every node a potential ``p``: edge i -> j carries flow only
    where ``p[j] - p[i] > c_ij``, and then ``flow[i, j] = (p[j] - p[i] - c_ij) / tau``, so ``p`` rises
    from ``a`` to ``b``.

    Parameters
    ----------
    graph : (n, n) scipy sparse array or matrix, numpy array or masked array
        Edge costs in SciPy's csgraph form: ``graph[i, j]`` is the cost of an edge from i to j. Every
        stored entry of a sparse graph is an edge, an explicit zero included; in a dense array, 0, inf
        and NaN are no edge, and so are the masked entries of a masked array. Costs must be finite and
        nonnegative; self-loops are ignored.
    a, b : (n,) array_like
        Source and sink masses, nonnegative. Only ``b - a`` enters the solve, and it must balance on
        every connected piece of the graph, since no mass crosses between pieces.
    tau : float, default 100
        Regularization strength (> 0); larger ``tau`` spreads the flow over more edges. 100 is
        DeltaTrace's value for costs up to 1 and masses that total 1; for masses that total M, use
        ``tau / M``.
    directed : bool, default False
        ``False`` uses every edge in both directions at the same cost; a pair stored both ways keeps the
        smaller cost. ``True`` uses each stored entry as a one-way edge, and the edge directions must let
        the mass reach ``b``.
    normalize_costs : bool, default True
        Divide the costs by the largest before solving, so ``tau`` does not depend on their scale; the
        cheapest paths stay the cheapest. ``False`` solves on the costs as given.
    verbose : int or bool, default 1
        Logging verbosity.

    Returns
    -------
    dict
        ``"flow"``: (n, n) scipy.sparse.csr_array, ``flow[i, j]`` the mass sent along i -> j (only edges
        that carry flow are stored). ``"potential"``: (n,) optimal potentials, in the units of the costs
        after normalization, with an arbitrary offset per connected piece. Nodes the flow leaves free sit
        at the midpoint of their optimal range. NaN where the optimum leaves a potential unbounded: every
        node in a piece without flow and, with ``directed=True``, nodes bounded on one side only.
        ``"pseudotime"``: the potentials shifted so that every piece starts at 0, then scaled to [0, 1]
        by the longest piece.
    """
    G = graph if sp.issparse(graph) or np.ma.isMaskedArray(graph) else np.asarray(graph, dtype=float)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"qrot_graph: graph must be square (n, n), got shape {G.shape}.")
    if not sp.issparse(G):      # SciPy's dense conventions: masked entries, or 0, inf and NaN, are no edge
        G = csg.csgraph_from_masked(G) if np.ma.isMaskedArray(G) else csg.csgraph_from_dense(G)
    G = sp.coo_array(sp.csr_array(G))
    n = G.shape[0]
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (n,) or b.shape != (n,):
        raise ValueError(f"qrot_graph: a and b need one entry per node ({n}), got {a.shape} and {b.shape}.")
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or (a < 0).any() or (b < 0).any():
        raise ValueError("qrot_graph: a and b must be finite and nonnegative.")
    if a.sum() <= 0 or not np.isclose(a.sum(), b.sum(), rtol=1e-6, atol=0):
        raise ValueError(f"qrot_graph: a and b must have the same positive total, got {a.sum()} and {b.sum()}.")
    b = b * (a.sum() / b.sum())     # equal totals to rounding, as the solver's balance check needs

    # edges: stored entries without self-loops; undirected keeps one cost per pair, used both ways
    keep = G.row != G.col
    tail, head, c = G.row[keep].astype(np.int64), G.col[keep].astype(np.int64), G.data[keep].astype(float)
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("qrot_graph: edge costs must be finite and nonnegative.")
    if not directed:
        u, v = np.minimum(tail, head), np.maximum(tail, head)
        o = np.lexsort((c, v, u))
        o = o[np.r_[True, (np.diff(u[o]) != 0) | (np.diff(v[o]) != 0)]] if o.size else o
        tail, head, c = np.r_[u[o], v[o]], np.r_[v[o], u[o]], np.r_[c[o], c[o]]
    if c.size == 0:
        raise ValueError("qrot_graph: the graph has no edges.")
    if normalize_costs and c.max() > 0:
        c = c / c.max()

    J, p = _solve_qrot(tail, head, c, b - a, tau)
    _log.detail(f"QROT solved: {c.size} edges, tau={tau}")

    flow = sp.csr_array((J, (tail, head)), shape=(n, n))
    flow.eliminate_zeros()
    # pseudotime: every piece starts at 0 (its offset is arbitrary), one scale for all pieces
    _, piece = csg.connected_components(sp.coo_array((np.ones(c.size), (tail, head)), shape=(n, n)),
                                        directed=False)
    start = np.full(piece.max() + 1, np.inf)
    np.fmin.at(start, piece, p)
    pt = p - start[piece]
    span = np.nanmax(pt) if np.isfinite(pt).any() else 0.0
    return {"flow": flow, "potential": p, "pseudotime": pt / span if span > 0 else pt}


# ---------------------------------------------------------------------------
# tl.compute_velocity_grid
# ---------------------------------------------------------------------------
@_log.stage("building the velocity grid")
def compute_velocity_grid(
    adata,
    basis: str = "umap",
    grid_size: tuple = (25, 25),
    k: int = 25,
    sigma: float | None = None,
    data_radius_factor: float = 0.7,
    mag_transform: str = "log1p",
    transform_scale: float | None = None,
    clip_quantile: float = 0.99,
    edge_threshold: float = 0,
    verbose: int | bool = 1,
    flow_key: str | None = None,
    key_added: str | None = None,
) -> None:
    """Turn the solved flow into a smoothed velocity field over an embedding.

    Gives each cell a velocity from the flow-weighted resultant of ALL its edges
    (net flow ``J[u->v] - J[v->u]``, so back-and-forth transport cancels and a
    branch point isn't snapped onto its single strongest edge), then smooths those
    per-cell vectors onto a regular grid by Gaussian-weighted kNN interpolation,
    ready for a quiver plot.

    Parameters
    ----------
    adata : anndata.AnnData
        Must hold ``uns["dt_flow"]`` (from :func:`qrot`) and embedding coordinates
        for ``basis``.
    basis : str, default "umap"
        Embedding name (looked up via ``X_<basis>`` / ``<basis>``).
    grid_size : tuple of int, default (25, 25)
        ``(nx, ny)`` number of grid cells along each axis.
    k : int, default 25
        Cells averaged per grid point when smoothing.
    sigma : float, optional
        Gaussian bandwidth; defaults to ``0.8 * hypot`` of grid spacing.
    data_radius_factor : float, default 0.7
        Suppress grid arrows whose nearest cell is farther than
        ``data_radius_factor * sigma``.
    mag_transform : {"log1p", "asinh", "none"}, default "log1p"
        Magnitude compression applied to arrow lengths.
    transform_scale : float, optional
        Scale for the magnitude transform; ``None`` uses the 80th percentile.
    clip_quantile : float, default 0.99
        Clip transformed magnitudes at this quantile.
    edge_threshold : float, default 0
        Minimum net flow for an edge to contribute to a cell's velocity.
    verbose : int or bool, default 1
        Logging verbosity.
    flow_key : str, optional
        ``uns`` key holding the solved flow. Overrides ``key_added`` when given.
    key_added : str, optional
        Suffix for the keys read and written, matching the one given to :func:`qrot`.

    Returns
    -------
    None
        Writes ``adata.uns["dt_grid"]`` (``Xg``/``Yg``/``U``/``V``/``mask``/``basis``/
        ``edges``) and ``adata.uns["dt_raw_arrows"]`` (per-cell diagnostics).
    """
    from scipy.spatial import cKDTree

    _sfx = _suffix(key_added)
    flow_key = flow_key or f"dt_flow{_sfx}"
    # Keep the user's transform_scale so the raw per-cell arrows below honor it too.
    user_transform_scale = transform_scale

    flow = adata.uns.get(flow_key)
    if flow is None:
        raise ValueError(
            f"No flow data in adata.uns['{flow_key}']. Run qrot() with adata_basis set "
            f"(and the same key_added) first."
        )
    edges = flow["edges"]
    J = flow["J"]

    coords = _resolve_basis(adata, basis)
    if not isinstance(coords, np.ndarray):       # e.g. a DataFrame in obsm
        coords = np.asarray(coords, dtype=float)

    obs_names = adata.obs_names.to_numpy()
    n = len(obs_names)
    J = np.asarray(J, dtype=float)
    nx_grid, ny_grid = grid_size

    # Per-cell velocity
    node_vec = np.zeros((n, 2), dtype=float)
    edges_arr = np.asarray(edges)
    if edges_arr.size:
        u_arr = edges_arr[:, 0].astype(np.int64)
        v_arr = edges_arr[:, 1].astype(np.int64)
        delta = (coords[v_arr] - coords[u_arr]).astype(np.float64)
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        unit = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 0)
        Jmat = sp.csr_matrix((J, (u_arr, v_arr)), shape=(n, n))
        net = J - np.asarray(Jmat[v_arr, u_arr]).ravel()  # subtract the reverse-edge flow
        if edge_threshold:
            net = np.where(np.abs(net) > edge_threshold, net, 0.0)
        w = net[:, None] * unit
        node_vec = np.column_stack(
            [np.bincount(u_arr, weights=w[:, j], minlength=n) for j in (0, 1)]
        )

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_edges = np.linspace(x_min, x_max, nx_grid + 1)
    y_edges = np.linspace(y_min, y_max, ny_grid + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xg, Yg = np.meshgrid(x_centers, y_centers)

    if sigma is None:
        sigma = 0.8 * np.hypot((x_max - x_min) / nx_grid, (y_max - y_min) / ny_grid)

    # One batched KD-tree query for every grid point at once
    tree = cKDTree(coords)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])  # (ny*nx, 2)
    k_eff = min(k, n)
    # k as a list of ranks keeps the (ny*nx, k_eff) shape even when k_eff == 1
    dists_flat, idxs_flat = tree.query(grid_pts, k=np.arange(1, k_eff + 1))

    # Gaussian-weighted average of the k nearest cells, per grid point
    within_flat = dists_flat[:, 0] <= data_radius_factor * sigma
    w_flat = np.exp(-(dists_flat**2) / (2 * sigma**2))  # (ny*nx, k_eff)
    U_flat = np.sum(w_flat * node_vec[idxs_flat, 0], axis=1)  # (ny*nx,)
    V_flat = np.sum(w_flat * node_vec[idxs_flat, 1], axis=1)
    W_flat = w_flat.sum(axis=1)

    mask_flat = (W_flat > 0) & within_flat
    U_flat = np.where(mask_flat, U_flat / np.maximum(W_flat, 1e-12), 0.0)
    V_flat = np.where(mask_flat, V_flat / np.maximum(W_flat, 1e-12), 0.0)

    U = U_flat.reshape(ny_grid, nx_grid)
    V = V_flat.reshape(ny_grid, nx_grid)
    mask = mask_flat.reshape(ny_grid, nx_grid)

    # Compress arrow lengths so the quiver plot is legible, and clip outliers
    mag = np.hypot(U, V)
    mag_pos = mag[mask & (mag > 0)]

    if mag_pos.size > 0:
        if transform_scale is None:
            transform_scale = max(float(np.quantile(mag_pos, 0.80)), 1e-12)

        if mag_transform == "asinh":
            mag_t = np.arcsinh(mag / transform_scale)
        elif mag_transform == "log1p":
            mag_t = np.log1p(mag / transform_scale)
        elif mag_transform == "none":
            mag_t = mag.copy()
        else:
            raise ValueError("mag_transform must be 'asinh', 'log1p', or 'none'")

        if clip_quantile is not None:
            cap = max(
                float(np.quantile(mag_t[mask & (mag_t > 0)], clip_quantile)), 1e-12
            )
            mag_t = np.minimum(mag_t, cap)

        U_plot = np.zeros_like(U)
        V_plot = np.zeros_like(V)
        dir_mask = mask & (mag > 0)
        U_plot[dir_mask] = (U[dir_mask] / mag[dir_mask]) * mag_t[dir_mask]
        V_plot[dir_mask] = (V[dir_mask] / mag[dir_mask]) * mag_t[dir_mask]
        arrow_mask = dir_mask
    else:
        U_plot = np.zeros_like(U)
        V_plot = np.zeros_like(V)
        arrow_mask = np.zeros((ny_grid, nx_grid), dtype=bool)

    _log.detail(
        f"basis '{basis}', {nx_grid}x{ny_grid} grid, "
        f"{int(arrow_mask.sum())} arrows from {len(J)} flow edges"
    )

    adata.uns[f"dt_grid{_sfx}"] = {
        "Xg": Xg,
        "Yg": Yg,
        "U": U_plot,
        "V": V_plot,
        "mask": arrow_mask,
        "basis": basis,
        "edges": np.array(edges, dtype=np.int64)
        if len(edges)
        else np.empty((0, 2), dtype=np.int64),
    }

    # Unsmoothed per-cell arrows (diagnostic; stored alongside the grid)
    raw_mag = np.linalg.norm(node_vec, axis=1)
    raw_valid = raw_mag > 0

    if raw_valid.any():
        raw_pos = raw_mag[raw_valid]
        raw_ts = (max(float(user_transform_scale), 1e-12) if user_transform_scale is not None
                  else max(float(np.quantile(raw_pos, 0.80)), 1e-12))
        if mag_transform == "asinh":
            raw_mag_t = np.arcsinh(raw_mag / raw_ts)
        elif mag_transform == "log1p":
            raw_mag_t = np.log1p(raw_mag / raw_ts)
        else:
            raw_mag_t = raw_mag.copy()
        if clip_quantile is not None:
            raw_cap = max(
                float(np.quantile(raw_mag_t[raw_valid], clip_quantile)), 1e-12
            )
            raw_mag_t = np.minimum(raw_mag_t, raw_cap)
        raw_U = np.where(
            raw_valid, (node_vec[:, 0] / np.maximum(raw_mag, 1e-12)) * raw_mag_t, 0.0
        )
        raw_V = np.where(
            raw_valid, (node_vec[:, 1] / np.maximum(raw_mag, 1e-12)) * raw_mag_t, 0.0
        )
    else:
        raw_U = np.zeros(n)
        raw_V = np.zeros(n)

    adata.uns[f"dt_raw_arrows{_sfx}"] = {
        "X": coords[:, 0],
        "Y": coords[:, 1],
        "U": raw_U,
        "V": raw_V,
        "mask": raw_valid,
        "basis": basis,
    }


# ---------------------------------------------------------------------------
# tl.trajectory  (the one-call path)
# ---------------------------------------------------------------------------
def _grid_unusable(adata, basis):
    """Why obsm's ``basis`` embedding cannot carry the velocity grid (finite 2-D
    coordinates, one row per cell), or None if it can. KeyError if it is absent."""
    try:
        X = np.asarray(_resolve_basis(adata, basis), dtype=float)
    except (TypeError, ValueError) as exc:
        return f"not numeric ({exc})"
    if X.ndim != 2 or X.shape != (adata.n_obs, 2):
        return f"of shape {X.shape}, not ({adata.n_obs}, 2)"
    if not np.isfinite(X).all():
        return "not finite everywhere"
    return None


def _velocity_basis(adata, basis):
    """The embedding trajectory() draws the velocity grid on, checked before any solve:
    ``basis`` itself, or with ``None`` the tissue (``[X_]spatial``) when present and usable,
    else UMAP."""
    if basis is not None:
        why = _grid_unusable(adata, basis)       # KeyError now, not after the solve
        if why:
            raise ValueError(
                f"obsm embedding {basis!r} is {why}; the velocity grid needs finite 2-D "
                f"coordinates. Pass another basis, or compute_velocity=False."
            )
        return basis
    skipped = []
    for cand in ("spatial", "umap"):
        if f"X_{cand}" not in adata.obsm and cand not in adata.obsm:
            continue
        why = _grid_unusable(adata, cand)
        if why is None:
            if skipped:
                warnings.warn(
                    f"Drawing the velocity grid on {cand!r}: obsm {skipped[0]} "
                    f"(pass basis= to choose explicitly).",
                    UserWarning, skip_file_prefixes=(_PKG_DIR,),
                )
            return cand
        skipped.append(f"{cand!r} is {why}")
    if skipped:
        raise ValueError(
            f"No usable embedding for the velocity grid: obsm {'; '.join(skipped)}. Pass "
            f"basis=<a finite 2-D obsm embedding>, or compute_velocity=False."
        )
    raise KeyError(
        "No embedding for the velocity grid (no 'spatial' or 'X_umap' in obsm): pass "
        "basis=, or compute_velocity=False."
    )


class _TrajectoryResult(dict):
    """What :func:`trajectory` returns. ``["graph"]``, the endpoints result's key before
    0.1.0, still reads ``["endpoints"]``, with a FutureWarning."""

    def __missing__(self, key):
        if key == "graph":
            warnings.warn(
                'trajectory()["graph"] is deprecated and will be removed; use '
                '["endpoints"] instead.',
                FutureWarning,
                stacklevel=2,
            )
            return self["endpoints"]
        raise KeyError(key)

    def __repr__(self):
        ep = self.get("endpoints") or {}
        n_src, n_snk = len(ep.get("source_cells", [])), len(ep.get("sink_cells", []))
        parts = [f"{n_src} source and {n_snk} sink cells" if n_snk
                 else f"{n_src} source cells, every other cell absorbing"]
        lin = self.get("lineages")
        if lin is not None:
            parts.append(f"lineages {list(lin.get('names', []))}")
        elif "terminality" in self:
            parts.append(f"no lineages assigned; terminality ranks {len(self['terminality'])} "
                         f"clusters")
        return f"<trajectory result: {'; '.join(parts)}; keys {list(self)}>"


def _trajectory_graph(adata, graph):
    """The graph trajectory() runs on: ``graph`` itself, else the intersection graph when
    it is stored, else the only graph there is (e.g. ``pp.graphs(adata)`` without a
    distance matrix stores just ``"expr"``)."""
    if graph is not None:
        return graph
    if "neighbors_intr_distances" in adata.obsp:
        return "intr"
    graph = _resolve_graph(adata, None)
    _log.detail(f"graph=None: no 'intr' graph stored, running on '{graph}'")
    return graph


@_renamed_kwargs(tag="graph", reconnect_key=("reconnect_graph", _graph_from_key))
@_log.container()
def trajectory(
    adata,
    source_clusters,
    sink_clusters=None,
    *,
    cluster_key: str = "clusters",
    graph: str | None = None,
    reconnect_graph: str = "expr",
    basis: str | None = None,
    p_end: float = 0.15,
    p_basis: str = "cluster",
    alloc_power: float = 1.0,
    sampling_mode="centered",
    directed_metric: str = "cost",
    tau: float = 100.0,
    seed: int = 42,
    edge_threshold: float = 0,
    compute_velocity: bool = True,
    omega_term: float | None = None,
    n_lineages: int | None = None,
    sink_weights: str = "uniform",
    verbose: int | bool = 1,
    key_added: str | None = None,
) -> dict:
    """Run the whole trajectory pipeline in one call.

    Composes the public stages: :func:`endpoints` -> :func:`qrot` ->
    :func:`compute_velocity_grid` -> :func:`lineages`. Those four document the
    arguments below; this wrapper only forwards them, and each stays callable on
    its own for when you need to vary one.

    The defaults (``p_end=0.15``, ``p_basis="cluster"``, ``sampling_mode="centered"``,
    ``tau=100``), with :func:`deltatrace.pp.graphs`' ``alpha=0.3, p_keep=0.3``, are the
    configuration selected for the intersection graph in the DeltaTrace benchmark
    (scRNA-seq with a paired ST section). The benchmark did not cover single-cell-
    resolution ST.

    Parameters
    ----------
    adata : anndata.AnnData
        Object carrying the graph(s), e.g. from :func:`deltatrace.pp.graphs`. Must
        hold ``obs[cluster_key]`` and, for the velocity step, an embedding (see
        ``basis``).
    source_clusters, sink_clusters : sequence of str, or None
        Emitting and absorbing cluster labels, as for :func:`endpoints`.
        ``sink_clusters=None`` runs SOURCE-ONLY: every non-source cell absorbs, and the
        clusters are ranked by how terminal the flow makes them (printed, and returned as
        ``"terminality"``). Lineages are then assigned only if ``omega_term`` or
        ``n_lineages`` cuts that ranking; otherwise pick the fates from it and run again
        with them as ``sink_clusters``, which gives the better trajectory.
    graph : str, optional
        Which stored graph to run on (``neighbors_<graph>``): ``"intr"`` (intersection)
        or ``"expr"`` from :func:`deltatrace.pp.graphs`, or a name given to
        :func:`deltatrace.pp.integrate` via ``key_added``. ``None`` takes ``"intr"`` when
        it is stored, else the only graph there is.
    cluster_key, p_end, p_basis, alloc_power, sampling_mode,
    directed_metric, seed, sink_weights
        Forwarded to :func:`endpoints`, which documents them (the defaults here are
        the same).
    tau, reconnect_graph
        Forwarded to :func:`qrot`.
    basis : str, optional
        Embedding for the velocity grid; forwarded to :func:`compute_velocity_grid`.
        ``None`` uses the tissue (``obsm['X_spatial']`` or ``obsm['spatial']``) when
        present as finite 2-D coordinates, else ``X_umap``. Checked before the solve.
    compute_velocity : bool, default True
        Run :func:`compute_velocity_grid` at all (it needs the ``basis`` embedding).
    edge_threshold, omega_term, n_lineages
        Forwarded to :func:`lineages`. ``omega_term``/``n_lineages`` are the two
        mutually exclusive source-only terminal cuts, meaningful only when
        ``sink_clusters is None``.
    verbose : int or bool, default 1
        Forwarded to every stage.
    key_added : str, optional
        Namespace this run's results. ``None`` writes the flat keys
        (``obs['pseudotime']``, ``obsm['path_cellweights']``, …); given a name,
        every per-run key gains a ``_<key_added>`` suffix, letting two runs
        coexist in one object. Graph keys (``neighbors_<graph>_*``) are scoped by
        ``graph`` instead and are never suffixed.

    Returns
    -------
    dict
        ``{"endpoints": <endpoints result>, "lineages": <lineages result>}``.
        Pseudotime, velocity and the lineage keys are written into ``adata`` in place. A
        source-only run without a cut returns ``"lineages": None`` and adds
        ``"terminality"``, the ranking to choose the fates from.
    """
    # up front: a bad terminal-cut combination or a missing embedding must not cost the
    # caller a full solve.
    _check_terminal_cut(sink_clusters, omega_term, n_lineages)
    source_clusters = _as_labels(source_clusters, "source_clusters")
    if sink_clusters is not None:
        sink_clusters = _as_labels(sink_clusters, "sink_clusters")
    _check_labels(adata, cluster_key, source_clusters=source_clusters,
                  sink_clusters=sink_clusters)
    graph = _trajectory_graph(adata, graph)
    if compute_velocity:
        basis = _velocity_basis(adata, basis)
    gr = endpoints(
        adata, adata, source_clusters, sink_clusters,
        p_end=p_end, sampling_mode=sampling_mode,
        cluster_key=cluster_key, graph=graph, seed=seed, sink_weights=sink_weights,
        verbose=verbose, directed_metric=directed_metric, p_basis=p_basis,
        alloc_power=alloc_power,
    )
    qrot(
        gr, tau=tau, adata_basis=adata, adata_clust=adata, reconnect=True,
        reconnect_graph=reconnect_graph, graph=graph, verbose=verbose,
        key_added=key_added,
    )
    if compute_velocity:
        compute_velocity_grid(adata, basis=basis, verbose=verbose,
                              key_added=key_added)
    if sink_clusters is None and omega_term is None and n_lineages is None:
        # Source-only discovery: rank the clusters and leave the choice of fates to the
        # caller. A default cut (say, the single most terminal cluster) would silently
        # merge the fates of any branching system.
        rank = dict(cluster_key=cluster_key, source_clusters=source_clusters,
                    edge_threshold=edge_threshold, key_added=key_added)
        tab = terminality(adata, show=False, **rank)
        if len(tab) == 0:
            raise ValueError(
                "Source-only run: no cluster receives flow from another cluster, so there "
                "are no fates to rank. Check source_clusters and cluster_key."
            )
        if _log.resolve(verbose) > 0:
            terminality(adata, show=True, **rank)
        omega, _ = suggest_omega(tab["term_score"].to_numpy())
        picked = [str(c) for c in tab.index[tab["term_score"] >= omega]]
        call = (f"dt.tl.lineages(adata=adata, source_clusters={[str(c) for c in source_clusters]}"
                + (f", cluster_key={cluster_key!r}" if cluster_key != "clusters" else "")
                + f", n_lineages={len(picked)}" + (f", key_added={key_added!r}" if key_added else "") + ")")
        warnings.warn(
            f"Source-only run: no lineages assigned. Rerun with sink_clusters={picked}, or call {call} "
            "on this flow.",
            UserWarning,
            skip_file_prefixes=(_PKG_DIR,),
        )
        return _TrajectoryResult(endpoints=gr, lineages=None, terminality=tab)
    lin = lineages(
        adata, source_clusters, sink_clusters, cluster_key=cluster_key,
        omega_term=omega_term, n_lineages=n_lineages, edge_threshold=edge_threshold,
        verbose=verbose, key_added=key_added,
    )
    return _TrajectoryResult(endpoints=gr, lineages=lin)
