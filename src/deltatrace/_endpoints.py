"""tl.endpoints: source/sink sampling and the demand vector."""

from __future__ import annotations

import hashlib
import math
import random as _random
import warnings

import numpy as np
import pandas as pd

from . import _log
from ._utils import (
    _PKG_DIR,
    _as_labels,
    _build_cell_graph,
    _check_labels,
    _resolve_graph,
)


# ---------------------------------------------------------------------------
# tl.endpoints
# ---------------------------------------------------------------------------
def _cluster_rng(seed, role, cluster) -> _random.Random:
    """Fresh random stream keyed by (seed, role, cluster), independent of draw order."""
    key = f"{seed}|{role}|{cluster}".encode()
    # hashlib, not the builtin hash(): that one is salted per process by PYTHONHASHSEED, so
    # the same arguments would sample different cells in a fresh interpreter.
    return _random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


@_log.stage("sampling source and sink endpoints")
def endpoints(
    adata,
    source_clusters,
    sink_clusters=None,
    p_end: float = 0.15,
    *,
    p_basis: str = "cluster",
    alloc_power: float = 1.0,
    sampling_mode: str | dict[str, str] = "centered",
    directed_metric: str = "cost",
    cluster_key: str = "clusters",
    graph: str | None = None,
    seed: int = 42,
    sink_weights: str = "uniform",
    verbose: int | bool = 1,
) -> dict:
    """Sample source and sink cells and construct the flow demand vector for QROT.

    Extracts the largest connected component (LCC) of the specified graph,
    samples representative cells from declared source and sink clusters according
    to `p_end` and `p_basis`, and sets up the balanced flow demand:

        f[source] = -1 / n_source
        f[sink]   = +1 / n_sink

    Also annotates `adata.obs["endpoint_<graph>"]` with categorical labels
    (``"source"``, ``"intermediate"``, ``"sink"``).

    Parameters
    ----------
    adata : anndata.AnnData or str
        AnnData containing graph connectivity in ``.obsp`` and cluster labels in
        ``.obs[cluster_key]``, or path to an ``.h5ad`` file.
    source_clusters : sequence of str
        Cluster labels designating flow progenitors (sources).
    sink_clusters : sequence of str, optional
        Cluster labels designating flow terminals (sinks). If None, runs in
        source-only mode where all non-source LCC cells act as absorbers.
    p_end : float, default 0.15
        Fraction of cells sampled as endpoints.
    p_basis : {"cluster", "lcc"}, default "cluster"
        Sampling quota basis. ``"cluster"`` allocates `p_end` of each cluster's
        pool (at least one cell per cluster). ``"lcc"`` allocates `p_end` of the
        entire LCC divided equally across clusters.
    alloc_power : float, default 1.0
        Cluster quota scaling exponent under ``p_basis="cluster"``. ``1.0`` is
        proportional to cluster size, ``0.0`` is equal allocation.
    sampling_mode : str or dict, default "centered"
        Sampling strategy: ``"centered"`` (closest to cluster medoid) or
        ``"random"``, or a dictionary specifying ``{"source": ..., "sink": ...}``.
    directed_metric : {"cost", "hops"}, default "cost"
        Metric used for geodesic distances in centered sampling.
    cluster_key : str, default "clusters"
        Column in ``adata.obs`` containing cluster labels.
    graph : str, optional
        Graph name to sample on (e.g. ``"intr"`` or ``"expr"``). Auto-detected if
        only one integrated graph is present.
    seed : int, default 42
        Random seed for sampling reproducibility.
    sink_weights : {"uniform", "cluster_balanced"}, default "uniform"
        In source-only mode, how demand is distributed across non-source cells.
    verbose : int or bool, default 1
        Verbosity level (0: silent, 1: stages, 2: detailed progress).

    Returns
    -------
    dict
        Flow problem dictionary containing:
        - ``"G"``: :class:`CellGraph` containing graph structure and node mapping.
        - ``"cost_vec"``: edge costs aligned with `G.edge_array`.
        - ``"f"``: balanced demand vector of shape ``(n_cells,)``.
        - ``"source_cells"``, ``"sink_cells"``: lists of sampled cell names.
        - ``"source_indices"``, ``"sink_indices"``: arrays of sampled cell indices.
        - ``"sink_cluster_map"``: mapping from sink cell indices to cluster names.
        - ``"disconnected_info"``: summary of off-LCC cells.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> gr = dt.tl.endpoints(
    ...     adata,
    ...     source_clusters=["progenitor"],
    ...     sink_clusters=["fate_A", "fate_B"],
    ...     p_end=0.15,
    ... )
    >>> "G" in gr and "f" in gr
    True
    """
    import scanpy as sc

    if isinstance(adata, str):
        adata = sc.read_h5ad(adata)

    if seed is None:
        # `_cluster_rng` keys on the seed's *string*, so None would freeze every draw on
        # the literal "None" -- draw and log a fresh seed instead, replayable after the fact.
        seed = _random.Random().randrange(2**32)
        _log.detail(f"seed=None: drew {seed}")
    np.random.seed(seed)
    # Left for _split_quota only: every SAMPLING draw comes from a per-(role, cluster)
    # stream built by _cluster_rng, so no cluster inherits the state its neighbors left.
    _random.seed(seed)

    # Resolve per-role sampling modes.
    _valid_source = {"random", "centered"}
    _valid_sink = {"random", "centered", "directed"}
    if isinstance(sampling_mode, str):
        if sampling_mode == "directed":
            source_mode, sink_mode = "centered", "directed"
        elif sampling_mode in ("random", "centered"):
            source_mode = sink_mode = sampling_mode
        else:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode!r}")
    elif isinstance(sampling_mode, dict):
        unknown = set(sampling_mode) - {"source", "sink"}
        if unknown:
            raise ValueError(
                f"sampling_mode dict keys must be 'source' and/or 'sink', got {sorted(unknown)}."
            )
        source_mode = sampling_mode.get("source", "centered")
        sink_mode = sampling_mode.get("sink", "centered")
    else:
        raise TypeError(
            f"sampling_mode must be str or dict, got {type(sampling_mode).__name__}"
        )
    if source_mode not in _valid_source:
        raise ValueError(
            f"Invalid source sampling mode: {source_mode!r}. "
            f"Source supports {_valid_source} only ('directed' is sink-only)."
        )
    if sink_mode not in _valid_sink:
        raise ValueError(
            f"Invalid sink sampling mode: {sink_mode!r}. Must be one of {_valid_sink}."
        )
    # Checked regardless of mode -- a misspelling is a caller error, not a no-op --
    # even where a valid value is silently inert (like sink_weights under source+sink).
    if directed_metric not in ("cost", "hops"):
        raise ValueError(
            f"Invalid directed_metric: {directed_metric!r}. Must be 'cost' or 'hops'."
        )
    hop_metric = directed_metric == "hops"

    if p_basis not in ("lcc", "cluster"):
        raise ValueError(
            f"p_basis must be 'lcc' or 'cluster', got {p_basis!r}."
        )

    # Checked under either basis, like directed_metric: a bad exponent is a caller error
    # even where it is inert. bool is an int in Python, but True is not an exponent.
    if (isinstance(alloc_power, bool)
            or not isinstance(alloc_power, (int, float, np.integer, np.floating))
            or not 0.0 <= float(alloc_power) <= 1.0):
        raise ValueError(
            f"alloc_power must be a number in [0, 1] (0 = equal split, 1 = proportional), "
            f"got {alloc_power!r}."
        )
    alloc_power = float(alloc_power)

    source_clusters = _as_labels(source_clusters, "source_clusters")
    if sink_clusters is not None:
        sink_clusters = _as_labels(sink_clusters, "sink_clusters")
    _check_labels(adata, cluster_key, source_clusters=source_clusters,
                  sink_clusters=sink_clusters)

    graph = _resolve_graph(adata, graph)

    import scipy.sparse.csgraph as csg

    # Build the bidirectional cell graph once, as scipy arrays (see CellGraph)
    G_full = _build_cell_graph(adata, graph=graph)
    name_to_idx = G_full.graph["name_to_idx"]
    idx_to_name = G_full.graph["idx_to_name"]
    # Every geodesic below runs on this one cost matrix. It is symmetric, so directed=True
    # walks the same edges the old undirected view of the DiGraph did.
    _cost_csr = G_full.cost
    n_nodes = G_full.n

    _log.detail(f"Full graph: {n_nodes} nodes, {G_full.m} directed edges")

    # Largest connected component (LCC), as _off_lcc_mask picks it. scipy numbers the
    # components by their lowest cell, so argmax gives a size tie to the component holding
    # the lowest cell index, as the old sorted(nx.connected_components(...)) did.
    n_comp, comp_labels = csg.connected_components(_cost_csr, directed=False)
    comp_sizes = np.bincount(comp_labels, minlength=n_comp)
    lcc_mask = comp_labels == comp_sizes.argmax()
    n_lcc = int(lcc_mask.sum())
    disconnected_indices = np.flatnonzero(~lcc_mask).tolist()

    _log.detail(
        f"LCC: {n_lcc} nodes, "
        f"{len(disconnected_indices)} disconnected, "
        f"{n_comp} components"
    )

    disconnected_info = {
        "indices": disconnected_indices,
        "names": [idx_to_name[i] for i in disconnected_indices],
        "component_sizes": sorted(comp_sizes.tolist(), reverse=True),
    }

    labels = adata.obs[cluster_key].astype(str)

    # Labeled cells present in the graph, in obs order: their cell index and label
    graph_names = pd.Index(list(name_to_idx))
    graph_idx = np.fromiter(name_to_idx.values(), dtype=np.int64, count=len(name_to_idx))
    pos = graph_names.get_indexer(labels.index)
    known = pos >= 0
    lab_cell = graph_idx[pos[known]]
    lab_name = labels.to_numpy()[known]

    def _get_eligible(clusters):
        out = {}
        in_lcc = lcc_mask[lab_cell]
        for c in clusters:
            # Python ints, as the samplers' rng draws index these lists
            idxs = lab_cell[in_lcc & (lab_name == c)].tolist()
            out[c] = idxs
            if _log.wants_detail():
                total = int((labels == c).sum())
                _log.detail(f"cluster '{c}': {total} total, {len(idxs)} in LCC")
        return out

    _log.detail("source eligibility:")
    elig_source = _get_eligible(source_clusters)
    if sink_clusters is not None:
        _log.detail("sink eligibility:")
        elig_sink = _get_eligible(sink_clusters)
    else:
        elig_sink = {}

    # Quota per role, sized off whichever population p_basis names.
    def _split_quota(role, N, clusters):
        """Split role total N equally across clusters; remainder to a random few,
        drawn from this role's own seeded stream. Returns {cluster: quota}."""
        base = N // len(clusters)
        rem = N % len(clusters)
        # sorted() plus a role-only stream: the remainder then depends only on seed and
        # cluster names, not on list order or what the other role already drew.
        shuffled = sorted(clusters)
        _cluster_rng(seed, "quota", role).shuffle(shuffled)
        quotas = {c: base for c in clusters}
        for c in shuffled[:rem]:
            quotas[c] += 1
        return quotas

    def _pool_quota(role, clusters, elig):
        """Size each cluster's quota from its own eligibility pool, weighted by
        pool_c ** alloc_power, floored at 1 and clamped to the pool. Deterministic
        (no randomness). Returns {cluster: quota}."""
        pools = {}
        for c in clusters:
            pool = len(elig.get(c, []))
            if pool == 0:
                raise ValueError(
                    f"{role} cluster '{c}': none of its {int((labels == c).sum())} cells "
                    f"are in the largest connected component of the '{graph}' graph. "
                    "Drop the cluster, or raise p_keep in pp.integrate."
                )
            pools[c] = pool
        proportional = {
            c: min(max(1, int(np.round(p_end * p))), p) for c, p in pools.items()
        }
        if alloc_power == 1.0:
            # Short circuit, not a formula that happens to agree: k=1 is bit-identical.
            return proportional
        # k=1's total is the budget, so p_end stays the size knob and k only
        # redistributes. Ties on the name: never on list order, pool size or a draw.
        budget = sum(proportional.values())
        weights = {c: float(p) ** alloc_power for c, p in pools.items()}
        w_total = sum(weights.values())
        exact = {c: budget * w / w_total for c, w in weights.items()}
        quotas = {c: math.floor(v) for c, v in exact.items()}
        names = list(pools)
        order = sorted(names, key=lambda c: (-(exact[c] - math.floor(exact[c])), c))
        for i in range(budget - sum(quotas.values())):
            quotas[order[i % len(order)]] += 1
        # Floor and clamp last, and any drift they cause is kept: re-balancing would need a
        # loop with no fixed point once every cluster sits at its clamp.
        return {c: min(max(1, quotas[c]), pools[c]) for c in names}

    if p_basis == "lcc":
        N = max(1, int(np.round(p_end * n_lcc)))
        src_quotas = _split_quota("source", N, source_clusters)
        snk_quotas = (
            {} if sink_clusters is None else _split_quota("sink", N, sink_clusters)
        )
        _log.detail(f"\nN per group: {N} (p_end={p_end}, LCC={n_lcc})")
        # Clamp an undersized cluster to what it has, rather than refuse -- at the cost
        # of the matched-totals guarantee p_basis='lcc' otherwise gives (hence the warning).
        for role, quotas, elig in (("source", src_quotas, elig_source),
                                   ("sink", snk_quotas, elig_sink)):
            for c, q in quotas.items():
                pool = len(elig.get(c, []))
                if not 0 < pool < q:
                    continue
                quotas[c] = pool
                warnings.warn(
                    f"p_end={p_end} asks {role} cluster {c!r} for {q} cells, but only "
                    f"{pool} are in the largest connected component; sampling all {pool}, "
                    "so the source and sink totals no longer match. Lower p_end to "
                    f"{p_end * pool / q:.3g}, or pass p_basis='cluster'.",
                    UserWarning,
                    skip_file_prefixes=(_PKG_DIR,),
                )
        # A zero quota drops the cluster silently further down (q == 0 is skipped), and
        # with it that cluster's lineage: say so. Only 'lcc' can do this ('cluster'
        # floors every quota at 1).
        for role, quotas in (("source", src_quotas), ("sink", snk_quotas)):
            empty = [c for c, q in quotas.items() if q == 0]
            if empty:
                warnings.warn(
                    f"p_end={p_end} gives {N} {role} cell(s) for {len(quotas)} {role} "
                    f"clusters, so {empty} get none"
                    + (" and will have empty lineages" if role == "sink" else "")
                    + f". Raise p_end to at least {len(quotas) / n_lcc:.3g}, or pass "
                    "p_basis='cluster'.",
                    UserWarning,
                    skip_file_prefixes=(_PKG_DIR,),
                )
    else:
        src_quotas = _pool_quota("Source", source_clusters, elig_source)
        snk_quotas = (
            {} if sink_clusters is None else _pool_quota("Sink", sink_clusters, elig_sink)
        )
        _log.detail(
            f"\nper-cluster quotas (p_end={p_end} of each role's own pool, "
            f"alloc_power={alloc_power})"
        )
    _log.detail(f"Source quotas: {src_quotas}")
    _log.detail(f"Sink quotas: {snk_quotas}")

    # Validate each role against ITS OWN pool: a cluster can be both source and sink.
    for role, quotas, elig in (("Source", src_quotas, elig_source),
                               ("Sink", snk_quotas, elig_sink)):
        for c, q in quotas.items():
            if q == 0:
                continue
            pool = elig.get(c, [])
            if len(pool) < q:
                raise ValueError(
                    f"{role} cluster '{c}' needs {q} cells, but only {len(pool)} of its "
                    f"{int((labels == c).sum())} are in the largest connected component of "
                    f"the '{graph}' graph. Lower p_end, or raise p_keep in pp.integrate.")

    # Helpers: exact graph medoid + centered sampling, both intra cluster. All geodesics
    # run on _cost_csr, in cost units or (unweighted) hops.
    def _medoid(eligible, rng):
        """Cluster cell with the smallest total distance to the others (subsampled
        above 400 eligible cells, using rng)."""
        eligible = list(eligible)
        if len(eligible) <= 1:
            return eligible[0]
        cells = eligible
        if len(eligible) > 400:  # bound cost on very large clusters
            # rng, not np.random (global and shared -- anything else drawing from it first
            # would move the medoid); sorted so ties in the argmin below break on cell index.
            cells = sorted(rng.sample(eligible, 400))
        idx = np.asarray(cells, dtype=np.int64)
        # Batched dijkstra calls, 50 sources each so no (400, n_obs) array is ever held;
        # `unweighted` is the only switch, so the cost branch is untouched
        # (breadth_first_order needs one call per source).
        sub = np.vstack([
            csg.dijkstra(_cost_csr, directed=True, indices=idx[s:s + 50],
                         unweighted=hop_metric)[:, idx]
            for s in range(0, idx.size, 50)
        ])
        sub[~np.isfinite(sub)] = 1e9  # unreachable pair -> finite sentinel
        # argmin keeps the first minimum, so ties resolve deterministically
        return int(idx[int(np.argmin(sub.sum(axis=1)))])

    def _degree_center(eligible):
        """Most-connected cell within the cluster (degree heuristic): neighbors inside
        the pool, counted off the cost matrix's rows. argmax keeps the first maximum,
        as max() over the pool did."""
        eligible = list(eligible)
        if len(eligible) <= 1:
            return eligible[0]
        idx = np.asarray(eligible, dtype=np.int64)
        in_pool = np.zeros(n_nodes, dtype=bool)
        in_pool[idx] = True
        R = _cost_csr[idx]
        owner = np.repeat(np.arange(idx.size), np.diff(R.indptr))
        counts = np.bincount(owner, weights=in_pool[R.indices], minlength=idx.size)
        return eligible[int(np.argmax(counts))]

    def _sssp(src):
        """Single-source geodesic lengths from src over the whole graph, in cost units
        or hops per directed_metric: an (n_obs,) float array, inf where unreachable."""
        return csg.dijkstra(_cost_csr, directed=True, indices=int(src),
                            unweighted=hop_metric)

    def _sssp_within(eligible, center, unweighted):
        """Geodesic lengths from center inside the subgraph the pool induces (the old
        G_und.subgraph(eligible)), aligned with eligible; inf where unreachable."""
        uniq, inv = np.unique(np.asarray(eligible, dtype=np.int64), return_inverse=True)
        sub = _cost_csr[uniq][:, uniq]
        at = int(np.flatnonzero(uniq == center)[0])  # center is always one of the pool
        d = csg.dijkstra(sub, directed=True, indices=at, unweighted=unweighted)
        return d[inv]

    _geodesic_cache = {}

    def _source_geodesics(centers):
        """(len(centers), n_obs) geodesics from each source center, from one batched
        Dijkstra that every directed sink cluster reuses (they all ask for the same)."""
        key = tuple(centers)
        if key not in _geodesic_cache:
            _geodesic_cache[key] = csg.dijkstra(
                _cost_csr, directed=True, indices=np.asarray(centers, dtype=np.int64),
                unweighted=hop_metric)
        return _geodesic_cache[key]

    def _find_sink_anchor(eligible, sink_center, source_centers, core_quantile=0.90):
        """Eligible sink cell farthest from its nearest source, restricted to the
        cluster core (within core_quantile of the medoid) to exclude outliers."""
        if not source_centers:
            return sink_center
        idx = np.asarray(eligible, dtype=np.int64)
        d_med = _sssp(sink_center)
        dm = d_med[idx]
        # The quantile runs over the pool's distinct cells, as it did over a {cell: d}
        # dict (a pool repeats a cell when obs_names repeat).
        d_cells = d_med[np.unique(idx)]
        finite = d_cells[np.isfinite(d_cells)]
        if not finite.size:
            return sink_center
        in_core = dm <= np.quantile(finite, core_quantile)
        if not in_core.any():
            in_core[:] = True                  # an empty core falls back to the whole pool
        core_pos = np.flatnonzero(in_core)
        # Each core cell's distance to its nearest source. Dijkstra never returns NaN, so
        # argmax picks the same first farthest cell max() over the core did.
        nearest = _source_geodesics(source_centers)[:, idx[core_pos]].min(axis=0)
        return eligible[int(core_pos[int(np.argmax(nearest))])]

    def _sample_centered_source(eligible, quota, rng, anchor=None):
        """Degree-centered source sampling with BFS hop-ring stratification (rng
        shuffles within each shell)."""
        center = anchor if anchor is not None else _degree_center(eligible)
        d = _sssp_within(eligible, center, unweighted=True)
        shells = {}
        for n, dn in zip(eligible, d.tolist(), strict=True):
            shells.setdefault(dn, []).append(n)
        out = []
        for dist in sorted(shells.keys()):
            shell = shells[dist]
            rng.shuffle(shell)
            out.extend(shell[: quota - len(out)])
            if len(out) >= quota:
                break
        return out[:quota], center

    def _sample_centered(eligible, quota, rng, anchor=None):
        """Medoid-centered sampling with shell stratification (medoid when anchor is
        None). Returns (sampled cell indices, center)."""
        center = anchor if anchor is not None else _medoid(eligible, rng)
        d = _sssp_within(eligible, center, unweighted=hop_metric)
        shells = {}
        for n, dn in zip(eligible, d.tolist(), strict=True):
            shells.setdefault(dn, []).append(n)
        out = []
        for dist in sorted(shells.keys()):
            shell = shells[dist]
            rng.shuffle(shell)
            out.extend(shell[: quota - len(out)])
            if len(out) >= quota:
                break
        return out[:quota], center

    # Sample sources
    source_cells_idx = []
    source_centers = []
    need_src_centers = sink_mode == "directed"
    for c, q in src_quotas.items():
        if q == 0:
            continue
        pool = elig_source[c]
        rng = _cluster_rng(seed, "source", c)
        if source_mode == "random":
            sampled = rng.sample(pool, q)
            if need_src_centers:
                source_centers.append(_degree_center(pool))
        else:  # "centered"
            sampled, center = _sample_centered_source(pool, q, rng)
            source_centers.append(center)
            _log.detail(f"Source center for '{c}': {idx_to_name[center]}")
        source_cells_idx.extend(sampled)

    # Sample sinks
    sink_cells_idx = []
    sink_cluster_map = {}  # cell index -> sink cluster name
    for c, q in snk_quotas.items():
        if q == 0:
            continue
        pool = elig_sink[c]
        rng = _cluster_rng(seed, "sink", c)
        if sink_mode == "random":
            sampled = rng.sample(pool, q)
        elif sink_mode == "centered":
            # Reuse the returned center
            sampled, center = _sample_centered(pool, q, rng)
            _log.detail(f"Sink center for '{c}': {idx_to_name[center]}")
        else:  # "directed"
            # Passing `anchor` stops _sample_centered re-drawing a medoid on a second
            # subsample of the same pool.
            sink_center = _medoid(pool, rng)
            anchor = _find_sink_anchor(pool, sink_center, source_centers)
            sampled, _ = _sample_centered(pool, q, rng, anchor=anchor)
            _log.detail(f"Sink anchor for '{c}': {idx_to_name[anchor]}")
        sink_cells_idx.extend(sampled)
        for _idx in sampled:
            sink_cluster_map[_idx] = c

    source_names = [idx_to_name[i] for i in source_cells_idx]
    sink_names = [idx_to_name[i] for i in sink_cells_idx]

    _log.detail(
        f"\nSampled {len(source_cells_idx)} source cells, "
        f"{len(sink_cells_idx)} sink cells"
    )

    # Cost vector aligned with G_full.edge_array (CSR storage order, the old G.edges() order)
    c = G_full.cost_vec.copy()

    # Demand vector f for B J = f
    f = np.zeros(n_nodes, dtype=np.float64)
    f[source_cells_idx] = -1.0 / len(source_cells_idx)
    if sink_clusters is not None:
        overlap = sorted(set(source_cells_idx) & set(sink_cells_idx))
        if overlap:
            # Sink assignment below overwrites source on these cells, unbalancing the
            # demand -- only reachable when one cluster is named in both roles; name it.
            per_cluster = {
                c: sum(1 for i in overlap if sink_cluster_map[i] == c)
                for c in sorted({sink_cluster_map[i] for i in overlap})
            }
            raise ValueError(
                f"{len(overlap)} cell(s) sampled as both a source and a sink, by cluster "
                f"{per_cluster}. Name each cluster in one role only, or lower p_end."
            )
        # Source + sink: mass is absorbed only at the sampled sink cells.
        f[sink_cells_idx] = 1.0 / len(sink_cells_idx)
        assert abs(f.sum()) < 1e-10, f"Flow imbalance: {f.sum()}"
    else:
        # Every LCC cell outside the source clusters absorbs, unlabeled ones included
        is_src = np.zeros(n_nodes, dtype=bool)
        is_src[lab_cell[np.isin(lab_name, source_clusters)]] = True
        rest = np.flatnonzero(lcc_mask & ~is_src)
        if rest.size == 0:
            raise ValueError("Source-only endpoints: no non-source cells in the LCC to absorb flow.")
        if sink_weights == "uniform":
            # Default: every absorber pulls equal mass, so a cluster's demand share is
            # its cell share (see param docs: per-cluster equalizing favors small clusters).
            w = np.ones(rest.size)
        elif sink_weights == "cluster_balanced":
            node_lab = np.full(n_nodes, "", dtype=object)    # "" for an unlabeled cell
            node_lab[lab_cell] = lab_name
            lab = node_lab[rest].astype(str)
            _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
            w = 1.0 / cnt[inv].astype(float)
        else:
            raise ValueError(f"sink_weights must be 'cluster_balanced' or 'uniform', got {sink_weights!r}")
        w = w / w.sum()
        f[rest] = w
        f[rest] -= f.sum() / rest.size
        _log.detail(
            f"source-only diffuse demand: {len(source_cells_idx)} source cells emit, "
            f"{rest.size} absorb (sink_weights={sink_weights})"
        )

    G_full.graph["source_indices"] = source_cells_idx
    G_full.graph["sink_indices"] = sink_cells_idx
    G_full.graph["source_names"] = source_names
    G_full.graph["sink_names"] = sink_names
    G_full.graph["disconnected_info"] = disconnected_info

    # The same three classes pl.endpoints draws, as an obs column scanpy can color by
    # (sc.pl.embedding(adata, color="endpoint_<graph>")). Categories are fixed so the
    # legend order and colors match pl.endpoints even when a class is empty.
    ep_class = np.full(adata.n_obs, "intermediate", dtype=object)
    ep_class[np.asarray(source_cells_idx, dtype=np.int64)] = "source"
    ep_class[np.asarray(sink_cells_idx, dtype=np.int64)] = "sink"
    adata.obs[f"endpoint_{graph}"] = pd.Categorical(
        ep_class, categories=["source", "intermediate", "sink"])
    adata.uns[f"endpoint_{graph}_colors"] = ["#3B4CC0", "lightgray", "#B40426"]

    return {
        "G": G_full,
        "cost_vec": c,
        "f": f,
        "source_cells": source_names,
        "sink_cells": sink_names,
        "source_indices": source_cells_idx,
        "sink_indices": sink_cells_idx,
        "disconnected_info": disconnected_info,
        "sink_cluster_map": sink_cluster_map,
    }


