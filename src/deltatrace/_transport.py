"""tl: the transport solve (qrot), the velocity grid, and trajectory() composing the stages."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

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
    import scipy.sparse.csgraph as csg

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

    name_to_idx = {name: i for i, name in enumerate(adata_clust.obs_names)}

    connected_names_list = [
        n for n in adata_clust.obs_names if n not in set(disc_names) and n in pt_dict
    ]
    connected_indices = [name_to_idx[n] for n in connected_names_list]

    dist_key = _graph_distances_key(adata_clust, reconnect_graph)
    if dist_key is None:
        raise ValueError(
            f"No graph {reconnect_graph!r} in adata_clust to reconnect off-LCC cells "
            f"through: neither obsp['neighbors_{reconnect_graph}_distances'] (a DeltaTrace "
            f"graph, e.g. reconnect_graph='expr' from pp.graphs) nor a scanpy graph stored "
            f"with key_added={reconnect_graph!r}. There is deliberately no fallback to "
            f"obsp['distances'], so a mis-specified graph cannot degrade reconnection "
            f"geodesics quietly."
        )
    D_clust = adata_clust.obsp[dist_key].tocsr()

    dist_sym = (D_clust + D_clust.T) / 2
    if dist_sym.diagonal().any():    # setdiag on an empty diagonal inserts zeros (scipy<1.13 warns)
        dist_sym.setdiag(0)
    dist_sym.eliminate_zeros()

    _targets = [n for n in sorted(disc_names) if n in name_to_idx]
    _log.detail(f"Reconnecting {len(_targets)} off-LCC cells from "
                f"{len(connected_indices)} LCC donors...")
    D_geodesic = csg.dijkstra(dist_sym, indices=[name_to_idx[n] for n in _targets])
    # column order is connected_indices, so argmin resolves a tie to the first donor
    D_to_donors = D_geodesic[:, connected_indices]

    pseudotime_full = dict(pt_dict)
    assigned = 0
    fallback = []

    for row, disc_name in enumerate(_targets):
        dists = D_to_donors[row]
        nearest_j = int(np.argmin(dists))
        nearest_dist = dists[nearest_j]

        if nearest_dist < np.inf:
            pseudotime_full[disc_name] = pseudotime_full[
                connected_names_list[nearest_j]
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
            f"component cannot be reached through reconnect_graph={reconnect_graph!r} "
            f"either: they got the connected cells' median pseudotime ({median_pt:.3g}) as "
            f"a placeholder, not an inferred position (they are listed in the endpoints "
            f"result's 'disconnected_info').",
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


@_renamed_kwargs(tag="graph", reconnect_key=("reconnect_graph", _graph_from_key))
@_log.stage("solving quadratically-regularized transport")
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
    solver_verbose: bool | None = None,
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
        Logging verbosity for DeltaTrace's own output (see ``solver_verbose``).
    solver_verbose : bool, optional
        Also forward the cvxpy/CLARABEL iteration log. Off by default.
    adata_basis : anndata.AnnData, optional
        If given, writes ``obs["pseudotime_raw"]`` (before reconnection),
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

    import cvxpy as cp

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
        # Min-max the edge costs into cost_range; a constant-cost graph maps to cost_range[0].
        _lo, _hi = cost_range
        _cmin, _cmax = c.min(), c.max()
        _span = _cmax - _cmin
        _scale = (_hi - _lo) / (_span if _span != 0 else 1.0)
        c = c * _scale + (_lo - _cmin * _scale)
        _log.detail(f"Costs normalized to [{c.min():.2e}, {c.max():.2e}]")

    # Incidence matrix B: column e holds -1 at its tail cell u and +1 at its head v.
    if node_map is None:
        us, vs = E[:, 0], E[:, 1]
    else:
        us = np.array([node_map[u] for u, _ in edges], dtype=np.int64)
        vs = np.array([node_map[v] for _, v in edges], dtype=np.int64)
    e = np.arange(m)
    B = sp.csr_matrix(
        (np.r_[-np.ones(m), np.ones(m)], (np.r_[us, vs], np.r_[e, e])),
        shape=(n, m),
    )

    f_arr = np.asarray(f, dtype=float)
    if f_arr.ndim != 1:
        # Explicit, rather than relying on numpy's element assignment to object: numpy
        # 1.26 only deprecates assigning a 1-element array into a scalar slot.
        raise ValueError(
            f"qrot: demand f must be a 1-D vector, got shape {f_arr.shape}."
        )
    f_vec = np.zeros(n, dtype=float)
    if node_map is None:
        # Element by element, as below, so both kinds of graph take the same path.
        for i in range(n):
            f_vec[i] = f_arr[i]
    else:
        for v, i in node_map.items():
            f_vec[i] = f_arr[v]

    J = cp.Variable(m, nonneg=True)
    obj = cp.Minimize(c @ J + (tau / 2) * cp.sum_squares(J))
    constraints = [f_vec == B @ J]
    prob = cp.Problem(obj, constraints)
    _solver_v = bool(solver_verbose)
    prob.solve(solver=cp.CLARABEL, verbose=_solver_v)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"QROT did not converge: status={prob.status}")
    _log.detail(f"solver: CLARABEL, status={prob.status}, objective={prob.value:.6g}")

    J_opt = np.array(J.value).ravel()

    if constraints[0].dual_value is not None:
        p = -np.array(constraints[0].dual_value).ravel()
    else:
        p = np.zeros(n)

    _log.detail(f"QROT solved: {m} edges, tau={tau}")
    _log.detail(f"  Flow range: [{J_opt.min():.4e}, {J_opt.max():.4e}]")
    _log.detail(f"  Potential range: [{p.min():.4e}, {p.max():.4e}]")

    idx_to_name = G.graph["idx_to_name"]
    if node_map is None:
        pt_raw = {idx_to_name[i]: p[i] for i in range(n)}
    else:
        pt_raw = {idx_to_name[node]: p[i] for node, i in node_map.items()}

    # Min-max the potentials onto [0, 1] (pseudotime)
    _vals = np.array(list(pt_raw.values()))
    _rng = max(_vals.max() - _vals.min(), 1e-12)
    pt_norm = {k: (v - _vals.min()) / _rng for k, v in pt_raw.items()}

    if adata_basis is not None:
        pt_raw_obs = np.array(
            [pt_norm.get(str(nm), np.nan) for nm in adata_basis.obs_names], dtype=float
        )
        adata_basis.obs[f"pseudotime_raw{_sfx}"] = pt_raw_obs

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
        pt_obs = np.array(
            [pt_full.get(str(nm), np.nan) for nm in adata_basis.obs_names], dtype=float
        )
        adata_basis.obs[f"pseudotime{_sfx}"] = pt_obs

    if adata_basis is not None:
        if node_map is None:
            # np.array of an empty edge list is shape (0,), not (0, 2): keep that
            edges_arr = E.copy() if m else np.array([], dtype=np.int64)
        else:
            edges_arr = np.array(edges, dtype=np.int64)
        adata_basis.uns[f"dt_flow{_sfx}"] = {
            "J": J_opt,
            "edges": edges_arr,
        }


# ---------------------------------------------------------------------------
# tl.compute_velocity_grid
# ---------------------------------------------------------------------------
@_log.stage("building the velocity grid")
def compute_velocity_grid(
    adata,
    basis: str = "umap",
    grid_size: tuple = (25, 25),
    k: int = 10,
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
    k : int, default 10
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
        np.add.at(node_vec, u_arr, net[:, None] * unit)

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
    dists_flat, idxs_flat = tree.query(grid_pts, k=k_eff)  # (ny*nx, k_eff)

    if k_eff == 1:  # tree.query returns 1-D when k=1
        dists_flat = dists_flat[:, None]
        idxs_flat = idxs_flat[:, None]

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
        "No embedding for the velocity grid: obsm holds neither 'spatial'/'X_spatial' "
        "nor 'X_umap'. Pass basis=<name of an obsm embedding>, or compute_velocity=False "
        "to skip the velocity step."
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
    p_end: float = 0.10,
    p_basis: str = "cluster",
    alloc_power: float = 1.0,
    sampling_mode="centered",
    directed_metric: str = "cost",
    tau: float = 100.0,
    seed: int = 42,
    edge_threshold: float = 1e-6,
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

    The defaults (``p_end=0.10``, ``p_basis="cluster"``, ``sampling_mode="centered"``,
    ``tau=100``), with :func:`deltatrace.pp.graphs`' ``alpha=0.3, p_keep=0.32``, are the
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
                "Source-only: no cluster receives flow from another cluster, so there is "
                "no terminality ranking to choose fates from. Check source_clusters and "
                "cluster_key, or pass the fates as sink_clusters."
            )
        printed = _log.resolve(verbose) > 0
        if printed:
            terminality(adata, show=True, **rank)
        omega, info = suggest_omega(tab["Term"].to_numpy())
        picked = [str(c) for c in tab.index[tab["Term"] >= omega]]
        warnings.warn(
            "Source-only run: pseudotime and flow are computed, but no lineages were "
            "assigned, because no terminal fates were chosen. The terminality ranking "
            f"({'printed; ' if printed else ''}result['terminality']) suggests {picked}"
            + (" (a clear gap)" if info["clear"] else " (no clear gap)")
            + ". Run again with the fates you choose as sink_clusters=[...] (recommended: "
            "sinks give the better trajectory), or assign lineages on this flow with "
            "dt.tl.lineages(adata, source_clusters, n_lineages=k"
            + (f", key_added={key_added!r}" if key_added else "") + ").",
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
