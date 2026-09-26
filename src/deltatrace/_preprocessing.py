"""pp: spatial distances (wasserstein), integrated graphs (integrate, graphs)."""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from . import _log
from ._utils import _MODE_GRAPHS, _PKG_DIR, _is_positional_index, _renamed_kwargs


# ---------------------------------------------------------------------------
# pp.wasserstein  (consumes a cell-to-spot mapping matrix, e.g. from Tangram)
# ---------------------------------------------------------------------------
def _pair_emd(idx_i, w_i, idx_j, w_j, D, max_dist):
    """EMD between cell i's and cell j's spot distributions, rescaled from the
    [0, 1] ground metric back to microns. Returns (finite, distance); distance
    is 0.0 and meaningless when finite is False. Shared kernel for wasserstein's
    serial loop and _compute_row_pairs, so the two paths can't drift apart."""
    import ot

    emd = ot.emd2(w_i, w_j, D[np.ix_(idx_i, idx_j)])
    finite = bool(np.isfinite(emd))
    return finite, float(emd * max_dist) if finite else 0.0


def _compute_row_pairs(
    i: int,
    nbrs_arr: np.ndarray,
    nbr_lens: np.ndarray,
    weights_arr: np.ndarray,
    D: np.ndarray,
    max_dist: float,
) -> tuple:
    """Worker for wasserstein's parallel path: EMD from cell i to every cell j > i,
    against the padded per-cell spot index/weight arrays. Returns (is, js, ds,
    n_dropped), the last being pairs whose EMD came back non-finite."""
    k_i = int(nbr_lens[i])
    if k_i == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), 0
    idx_i = nbrs_arr[i, :k_i]
    w_i = weights_arr[i, :k_i]
    n_cells = len(nbr_lens)
    buf_i, buf_j, buf_d = [], [], []
    n_dropped = 0
    for j in range(i + 1, n_cells):
        k_j = int(nbr_lens[j])
        if k_j == 0:
            continue
        idx_j = nbrs_arr[j, :k_j]
        w_j = weights_arr[j, :k_j]
        finite, d = _pair_emd(idx_i, w_i, idx_j, w_j, D, max_dist)
        if not finite:
            n_dropped += 1
            continue
        buf_i.append(i)
        buf_j.append(j)
        buf_d.append(d)
    if not buf_d:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), n_dropped
    return (
        np.array(buf_i, dtype=np.int32),
        np.array(buf_j, dtype=np.int32),
        np.array(buf_d, dtype=np.float32),
        n_dropped,
    )


@_log.stage("computing cell-cell Wasserstein distances")
def wasserstein(
    M,
    adata_sp: ad.AnnData,
    pxl_to_um: float = 1.0,
    mass_threshold: float = 0.9,
    verbose: int | bool = 1,
    n_jobs: int = -1,
) -> ad.AnnData:
    """Earth-Mover's distance between cells' spot-mapping distributions.

    Each cell is a distribution over spatial spots (the cell-to-spot mapping ``M``,
    e.g. from tangram); spot-spot Euclidean distance D is the ground metric:

        W1(a, b) = min_{P >= 0}  <D, P>
        s.t.                     P 1 = a,  P^T 1 = b

    Only the top ``mass_threshold`` fraction of each cell's mass is kept, which is
    what keeps the all-pairs cost tractable. Pairs whose EMD comes back non-finite
    are dropped and reported once as a ``RuntimeWarning``.

    Parameters
    ----------
    M : anndata.AnnData or str
        Cell-to-spot mapping ``(n_cells, n_spots)``, each row a distribution over
        spots. A path to an ``.h5ad`` file is also accepted. When both ``M.var_names``
        and ``adata_sp.obs_names`` are named, the spots are matched by name (a
        ValueError if they name different spots or repeat one); with a default
        0, 1, 2, ... index on either side, ``M``'s columns are read in ``adata_sp``'s order.
    adata_sp : anndata.AnnData
        Spatial reference; needs ``obsm["spatial"]`` of shape ``(n_spots, 2)``.
    pxl_to_um : float, default 1.0
        Pixel-to-micron scale.
    mass_threshold : float, default 0.9
        Per-cell cumulative-mass cutoff for which spots are kept and renormalized.
    verbose : int or bool, default 1
        Logging verbosity (0 silent, 2 most detailed).
    n_jobs : int, default -1
        Joblib workers; ``-1`` uses all cores. Every cell pair is solved, so this is
        the step that grows fastest with the number of cells. ``1`` runs serially in
        this process (float64 weights; the parallel path holds them as float32, so the
        two agree to float32 precision, relative ~1e-7). Starting the workers costs about
        a second, and joblib falls back to one process where it cannot spawn any.

    Returns
    -------
    anndata.AnnData
        Symmetric ``(n_cells, n_cells)`` sparse distance matrix (microns), indexed
        by ``M``'s cell names, for :func:`integrate` / :func:`graphs`.
    """
    import scanpy as sc

    if isinstance(M, str):
        M = sc.read_h5ad(M)

    # --- validate the mapping-side inputs ---------------------------------------
    if "spatial" not in adata_sp.obsm:
        raise KeyError(
            "adata_sp.obsm['spatial'] not found. The Wasserstein ground metric "
            "is built from the spatial spot coordinates of the reference used to "
            "produce the mapping M. Set adata_sp.obsm['spatial'] (shape "
            "(n_spots, 2)) before calling wasserstein()."
        )
    n_spots = adata_sp.n_obs
    if M.shape[1] != n_spots:
        raise ValueError(
            f"Mapping matrix M has {M.shape[1]} spot columns but adata_sp has "
            f"{n_spots} spots. M.X must be (n_cells, n_spots) aligned to adata_sp."
        )

    coords = np.asarray(adata_sp.obsm["spatial"])
    # M's columns and adata_sp's rows must be the same spots in the same order; with names
    # on both sides, match them by name rather than trust the order.
    m_spots = [str(s) for s in M.var_names]
    sp_spots = [str(s) for s in adata_sp.obs_names]
    if (m_spots != sp_spots and not _is_positional_index(m_spots)
            and not _is_positional_index(sp_spots)):
        if len(set(sp_spots)) != len(sp_spots) or len(set(m_spots)) != len(m_spots):
            raise ValueError(
                "M.var_names or adata_sp.obs_names repeats a spot name, so the mapping's "
                "columns cannot be matched to the spots by name. De-duplicate the names."
            )
        if set(m_spots) != set(sp_spots):
            raise ValueError(
                "M's spot columns (var_names) and adata_sp's spots (obs_names) name "
                f"different spots, e.g. {sorted(set(m_spots) - set(sp_spots))[:3]} vs "
                f"{sorted(set(sp_spots) - set(m_spots))[:3]}. Pass the spatial object the "
                "mapping was made on. If the two only label the same spots differently and "
                "are already in the same order, give M a default 0, 1, 2, ... var index to "
                "keep the positional reading."
            )
        pos = {nm: i for i, nm in enumerate(sp_spots)}
        coords = coords[[pos[nm] for nm in m_spots]]
        _log.detail("Matched M's spot columns to adata_sp's spots by name")
    coords = coords * pxl_to_um
    D_um = cdist(coords, coords, metric="euclidean")
    max_dist = float(np.max(D_um))
    D_scaled = (D_um / max_dist).astype(np.float32)
    del D_um

    X = M.X
    if sp.issparse(X):
        X = X.tocsr()

    # Build per-cell spot lists filtered to top mass_threshold fraction
    nbrs = []
    weights = []

    for i in range(X.shape[0]):
        if sp.issparse(X):
            indices = X[i].indices
            w = X[i].data
        else:
            indices = np.flatnonzero(X[i])
            w = X[i, indices]

        if w.size == 0:
            nbrs.append(np.array([], dtype=int))
            weights.append(np.array([], dtype=float))
            continue

        order = np.argsort(-w)
        w_sorted = w[order]
        idx_sorted = indices[order]

        # Cumulative mass cutoff
        cumsum = np.cumsum(w_sorted)
        k = np.searchsorted(cumsum, mass_threshold * cumsum[-1]) + 1

        top_idx = idx_sorted[:k]
        top_w = w_sorted[:k]
        top_w = top_w / top_w.sum()

        nbrs.append(top_idx)
        weights.append(top_w)

    n_cells = len(nbrs)
    total_pairs = n_cells * (n_cells - 1) // 2

    if _log.wants_detail():
        n_spots_per_cell = [len(idx) for idx in nbrs]
        _log.detail(
            f"{mass_threshold * 100:.0f}% mass-threshold spots per cell: "
            f"min {np.min(n_spots_per_cell)}, mean {np.mean(n_spots_per_cell):.1f}, "
            f"median {np.median(n_spots_per_cell):.0f}, max {np.max(n_spots_per_cell)}"
        )
    _log.detail(f"{n_cells} cells, {total_pairs} pairs (n_jobs={n_jobs})")

    # Pre-allocate output arrays
    ci = np.empty(total_pairs, dtype=np.int32)
    cj = np.empty(total_pairs, dtype=np.int32)
    dists = np.empty(total_pairs, dtype=np.float32)
    ptr = 0
    n_dropped = 0

    if n_jobs == 1:
        for i in range(n_cells):
            if i % 100 == 0:
                _log.detail(f"processing cell {i}/{n_cells}")
            idx_i = nbrs[i]
            w_i = weights[i]
            if w_i.size == 0:
                continue
            for j in range(i + 1, n_cells):
                idx_j = nbrs[j]
                w_j = weights[j]
                if w_j.size == 0:
                    continue
                finite, d = _pair_emd(idx_i, w_i, idx_j, w_j, D_scaled, max_dist)
                if finite:
                    dists[ptr] = d
                    ci[ptr] = i
                    cj[ptr] = j
                    ptr += 1
                else:
                    n_dropped += 1
    else:
        import joblib
        from joblib import Parallel, delayed

        nbr_lens = np.array([len(x) for x in nbrs], dtype=np.int32)
        max_k = int(nbr_lens.max()) if n_cells > 0 else 0
        nbrs_arr = np.full((n_cells, max_k), -1, dtype=np.int32)
        weights_arr = np.zeros((n_cells, max_k), dtype=np.float32)
        for i, (idx, w) in enumerate(zip(nbrs, weights, strict=True)):
            nbrs_arr[i, : len(idx)] = idx
            weights_arr[i, : len(w)] = w

        with tempfile.NamedTemporaryFile(suffix=".mmap", delete=False) as f:
            mmap_path = f.name
        try:
            joblib.dump((nbrs_arr, nbr_lens, weights_arr, D_scaled), mmap_path)
            nbrs_s, lens_s, weights_s, D_s = joblib.load(mmap_path, mmap_mode="r")

            gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
                delayed(_compute_row_pairs)(i, nbrs_s, lens_s, weights_s, D_s, max_dist)
                for i in range(n_cells)
            )
            for ri, rj, rd, rdrop in gen:
                n_dropped += rdrop
                n = len(rd)
                if n > 0:
                    ci[ptr : ptr + n] = ri
                    cj[ptr : ptr + n] = rj
                    dists[ptr : ptr + n] = rd
                    ptr += n
        finally:
            os.unlink(mmap_path)

    if n_dropped:
        warnings.warn(
            f"{n_dropped} of {total_pairs} cell pairs had a non-finite EMD and were "
            f"dropped from the Wasserstein graph (usually degenerate/empty spot "
            f"distributions). Their edges are simply absent.",
            RuntimeWarning,
            stacklevel=2,
        )

    ci = ci[:ptr]
    cj = cj[:ptr]
    dists = dists[:ptr]

    if _log.wants_detail():
        n_valid = len(dists)
        _log.detail(
            f"valid pairs: {n_valid}/{total_pairs} "
            f"({100 * n_valid / total_pairs:.2f}%)"
        )
        if n_valid > 0:
            _log.detail(
                f"distance (microns): min {dists.min():.4g}, mean {dists.mean():.4g}, "
                f"median {np.median(dists):.4g}, max {dists.max():.4g}"
            )

    # Build symmetric sparse distance matrix (microns)
    wd_upper = sp.coo_matrix(
        (dists, (ci, cj)), shape=(n_cells, n_cells), dtype=np.float32
    )
    wd_sparse = (wd_upper + wd_upper.T).tocsr()
    d_max = wd_sparse.data.max() if wd_sparse.data.size else 0.0

    cell_names = M.obs_names
    adata_wd = ad.AnnData(
        X=wd_sparse,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=cell_names),
    )
    adata_wd.uns["d_max"] = float(d_max)
    adata_wd.uns["mass_threshold"] = mass_threshold
    adata_wd.uns["cost"] = "euclidean"

    return adata_wd


# Per-dataset quantile of -ln(conn) clipping transport costs. No public knob: retune here.
_COST_CAP_QUANTILE = 0.99


def _align_distance_matrix(names_sc, names_dist, ext_dist):
    """Permute `ext_dist` into adata_sc's row order, matched against adata_dist's
    names (positional index on either side is read as "no identity", not compared).
    Raises ValueError on unresolvable mismatches or duplicate names.

    integrate() fuses ext_dist with the expression graph position by position, so a
    mismatch here would silently fuse the wrong cells rather than error.
    """
    from collections import Counter

    sc_names = [str(nm) for nm in names_sc]
    d_names = [str(nm) for nm in names_dist]
    if sc_names == d_names:
        return ext_dist
    if _is_positional_index(d_names):
        if not _is_positional_index(sc_names):
            warnings.warn(
                "adata_dist has no cell names (a default 0, 1, 2, ... index), so its rows "
                "are taken to follow adata's cell order, unchecked. Give it adata's cell "
                "names, e.g. ad.AnnData(X=D, obs=pd.DataFrame(index=adata.obs_names)), "
                "and it is matched to the cells by name instead.",
                UserWarning,
                skip_file_prefixes=(_PKG_DIR,),
            )
        return ext_dist
    if _is_positional_index(sc_names) and set(sc_names) != set(d_names):
        return ext_dist

    for who, names in (("adata_dist", d_names), ("adata_sc", sc_names)):
        counts = Counter(names)
        dupes = sorted(nm for nm, k in counts.items() if k > 1)[:3]
        if dupes:
            raise ValueError(
                f"{who}.obs_names repeats {dupes}, so the rows of adata_dist cannot be "
                "matched to adata_sc unambiguously. De-duplicate the names (see "
                "AnnData.obs_names_make_unique), or pass a distance object already in "
                "adata_sc's row order with a default index."
            )

    d_pos = {nm: i for i, nm in enumerate(d_names)}
    missing = [nm for nm in sc_names if nm not in d_pos]
    if missing:
        sc_set = set(sc_names)
        extra = [nm for nm in d_names if nm not in sc_set]
        raise ValueError(
            f"adata_sc and adata_dist name different cells: {len(missing)} of "
            f"adata_sc.obs_names are absent from adata_dist, e.g. {missing[:3]}"
            + (f"; adata_dist has e.g. {extra[:3]} instead" if extra else "")
            + ". They must describe the same cells. If the two objects simply label the "
            "same cells differently and adata_dist is already in adata_sc's row order, "
            "pass it with a default 0, 1, 2, ... index to keep the positional reading."
        )

    perm = np.array([d_pos[nm] for nm in sc_names], dtype=np.int64)
    n_moved = int((perm != np.arange(perm.size)).sum())
    _log.detail(
        f"Realigned adata_dist to adata_sc.obs_names ({n_moved} of {perm.size} rows moved)"
    )
    return ext_dist[perm][:, perm].tocsr()


def _blend_with_distances(adata, adata_dist, expr_conn, alpha, p_keep, mode):
    """The spatial half of :func:`integrate`: align and filter the distance matrix, then
    blend it with the expression connectivities. Returns the integrated connectivity
    matrix and the entries integrate() records about it in ``uns[...]["params"]``."""
    n_cells = expr_conn.shape[0]

    # Extract and filter distance matrix
    ext_dist = adata_dist.X.copy()
    if ext_dist.dtype != np.float64:
        ext_dist = ext_dist.astype(np.float64)
    if not sp.issparse(ext_dist):
        ext_dist = sp.csr_matrix(ext_dist)
    ext_dist = ext_dist.tocsr()

    if ext_dist.shape[0] != n_cells:
        raise ValueError(
            f"Cell count mismatch: sc={n_cells}, dist={ext_dist.shape[0]}"
        )
    if ext_dist.shape[1] != n_cells:
        raise ValueError(
            f"Distance matrix must be square over the cells: got {ext_dist.shape[0]}x"
            f"{ext_dist.shape[1]} for {n_cells} cells."
        )
    ext_dist = _align_distance_matrix(adata.obs_names, adata_dist.obs_names, ext_dist)
    if ext_dist.data.size == 0:
        raise ValueError("Distance matrix has no nonzero entries.")

    # A non-finite entry does not raise anywhere downstream: np.quantile returns nan, the
    # `> threshold` comparison is then False everywhere so nothing is filtered, ext_dist_max
    # is nan, and EVERY stored value of both output matrices comes out nan. Catch it here,
    # where the caller can still tell which matrix was bad.
    if not np.isfinite(ext_dist.data).all():
        n_bad = int((~np.isfinite(ext_dist.data)).sum())
        raise ValueError(
            f"Distance matrix has {n_bad} non-finite entries (NaN or inf). These do not "
            "raise downstream -- they propagate silently into every edge of the integrated "
            "graph. Common source: arccos of a dot product that floating point pushed just "
            "past 1.0. Clip the input before calling integrate()."
        )
    if (ext_dist.data < 0).any():
        raise ValueError(
            f"Distance matrix has {int((ext_dist.data < 0).sum())} negative entries. "
            "integrate() maps distance to similarity as 1 - d/d_max, which is only "
            "meaningful for d >= 0."
        )

    # The p_keep quantile runs over the STORED values, so a stored diagonal is counted as
    # n zero-distances and drags the threshold down -- silently retuning p_keep by an amount
    # that depends on how dense the caller's matrix is. A cell's distance to itself is never
    # an edge, so drop it before the quantile rather than after.
    if ext_dist.diagonal().any():
        _log.detail(f"Dropping {int((ext_dist.diagonal() != 0).sum())} stored diagonal entries")
        ext_dist = ext_dist.tolil()
        ext_dist.setdiag(0)
        ext_dist = ext_dist.tocsr()
        ext_dist.eliminate_zeros()
        if ext_dist.data.size == 0:
            raise ValueError("Distance matrix has no off-diagonal nonzero entries.")

    # The graph the prior defines is undirected, but nothing below symmetrizes the SUPPORT:
    # the intersection mask, and hence which edges survive, would otherwise depend on which
    # triangle the caller happened to fill in. (`maximum` is a no-op on an already-symmetric
    # matrix, so this cannot change a well-formed input.)
    _asym = abs(ext_dist - ext_dist.T)
    _asym.eliminate_zeros()
    if _asym.nnz:
        _log.detail(f"Symmetrizing distance matrix ({_asym.nnz} asymmetric entries)")
        ext_dist = ext_dist.maximum(ext_dist.T).tocsr()

    ext_dist_unfiltered_max = float(ext_dist.data.max())
    threshold = np.quantile(ext_dist.data, p_keep)

    ext_dist_filt = ext_dist.copy()
    ext_dist_filt.data[ext_dist_filt.data > threshold] = 0
    ext_dist_filt.eliminate_zeros()

    _log.detail(
        f"Distance filtering: kept {ext_dist_filt.nnz} / {ext_dist.nnz} edges "
        f"(p_keep={p_keep}, threshold={threshold:.4g})"
    )

    # Normalize to [0, 1]
    expr_conn_max = expr_conn.data.max()
    ext_dist_max = ext_dist_filt.data.max()

    expr_c_n = expr_conn.copy()
    expr_c_n.data = expr_c_n.data / expr_conn_max
    ext_d_n = ext_dist_filt.copy()
    ext_d_n.data = ext_d_n.data / ext_dist_max
    ext_c_n = ext_d_n.copy()
    ext_c_n.data = 1.0 - ext_c_n.data  # similarity = 1 - dist

    _log.detail(
        f"Normalization - expr_conn_max: {expr_conn_max:.4g}, "
        f"ext_dist_max: {ext_dist_max:.4g}"
    )

    # kNN side = connectivities only -> mask from the symmetric connectivity support
    expr_mask = expr_c_n.copy()
    expr_mask.data[:] = 1.0
    ext_mask = ext_d_n.copy()
    ext_mask.data[:] = 1.0
    intr_mask = expr_mask.multiply(ext_mask).tocsr()  # nonzero only at intersection

    # Blended connectivity at intersection positions
    intr_c = (
        expr_c_n.multiply(intr_mask) * (1 - alpha) + ext_c_n.multiply(intr_mask) * alpha
    )

    if mode == "intersection":
        int_conn_csr = intr_c.tocsr()

        _log.detail(
            f"Intersection: {intr_mask.nnz} edges "
            f"(from expr={expr_c_n.nnz}, dist={ext_d_n.nnz})"
        )

    elif mode == "expression":
        # expr-only positions: expression has a connectivity, ext does not.
        expr_only_c = expr_c_n - expr_c_n.multiply(intr_mask)
        expr_only_c.eliminate_zeros()
        expr_only_c_blend = expr_only_c * (1 - alpha)
        int_conn_csr = (intr_c + expr_only_c_blend).tocsr()

        n_both = intr_mask.nnz
        n_expr_only = expr_only_c.nnz

        _log.detail(f"Integration: {n_both} edges in both, {n_expr_only} expr-only")
        _log.detail(f"Integrated graph: {int_conn_csr.nnz} total edges")

    else:
        raise ValueError(
            f"mode must be 'intersection' or 'expression', got {mode!r}")

    if mode == "expression":
        info = {
            "alpha": alpha,
            "p_keep": p_keep,
            "expr_conn_max": float(expr_conn_max),
            "ext_dist_max": float(ext_dist_max),
            "n_edges_both": int(n_both),
            "n_edges_expr_only": int(n_expr_only),
        }
    else:  # intersection
        info = {
            "alpha": alpha,
            "p_keep": p_keep,
            "expr_conn_max": float(expr_conn_max),
            "ext_dist_max": float(ext_dist_max),
            "ext_dist_threshold_raw": float(threshold),
            "ext_dist_unfiltered_max": ext_dist_unfiltered_max,
            "n_edges_intersection": int(int_conn_csr.nnz),
            "n_edges_expr": int(expr_conn.nnz),
            "n_edges_dist_filtered": int(ext_dist_filt.nnz),
        }
    return int_conn_csr, info


# ---------------------------------------------------------------------------
# pp.integrate
# ---------------------------------------------------------------------------
def _integrate_message(a) -> str:
    graph = a.get("key_added") or _MODE_GRAPHS.get(a.get("mode"), a.get("mode"))
    return (f"building the '{graph}' graph"
            + (" from expression alone" if a.get("adata_dist") is None else ""))


@_renamed_kwargs(tag="key_added")
@_log.stage(_integrate_message)
def integrate(
    adata_sc,
    adata_dist=None,
    alpha: float = 0.3,
    p_keep: float = 0.32,
    mode: str = "intersection",
    neighbors_key: str = "neighbors",
    key_added: str | None = None,
    verbose: int | bool = 1,
) -> ad.AnnData:
    """Constrain an expression kNN graph by how far apart cells are in the tissue.

    Each edge gets a blended connectivity, and a transport cost derived from it:

        conn = alpha * spatial + (1 - alpha) * expression
        cost = clip(-ln(conn), floor, cap)

    ``mode="intersection"`` keeps only edges present in both graphs (can disconnect
    cells); ``mode="expression"`` keeps every expression edge, penalizing spatially
    implausible ones instead of dropping them.

    Parameters
    ----------
    adata_sc : anndata.AnnData or str
        Single-cell object with a KNN graph in ``.obsp``. Path to ``.h5ad`` accepted.
    adata_dist : anndata.AnnData or str, optional
        Distance object, ``X`` of shape ``(n_cells, n_cells)`` (e.g. from
        :func:`wasserstein`), aligned to ``adata_sc`` by ``obs_names``. Without cell
        names it is read in ``adata_sc``'s order (with a warning when ``adata_sc`` has
        cell names). ``None`` (no spatial
        input) builds the plain expression graph: ``mode="expression"`` only, and
        ``alpha``/``p_keep`` are ignored.
    alpha : float, default 0.3
        Spatial blending weight in ``[0, 1]``. ``0`` drops the spatial term: with
        ``mode="expression"`` the result is then the plain expression graph.
    p_keep : float, default 0.32
        Quantile of distance values kept when filtering the distance graph.
    mode : {"intersection", "expression"}, default "intersection"
        Integration mode (see above).
    neighbors_key : str, default "neighbors"
        Base key of the expression KNN graph in ``.obsp``/``.uns``.
    key_added : str, optional
        Name to store the graph under (``neighbors_<key_added>``), which downstream
        calls select with ``graph=``; defaults to ``"intr"``/``"expr"`` by mode.
    verbose : int or bool, default 1
        Logging verbosity.

    Returns
    -------
    anndata.AnnData
        Copy of ``adata_sc`` with the integrated graph under ``neighbors_<key_added>``
        (``obsp`` distances/connectivities, ``uns`` params); intersection mode also
        adds ``obs``/``uns["disconnected_<key_added>"]``. Unlike
        ``sc.pp.neighbors(key_added=)``, the name gets the ``neighbors_`` prefix, so scanpy
        reads the graph with ``neighbors_key="neighbors_<key_added>"``.
    """
    import scanpy as sc

    if isinstance(adata_sc, str):
        adata_sc = sc.read_h5ad(adata_sc)
    if isinstance(adata_dist, str):
        adata_dist = sc.read_h5ad(adata_dist)
    if mode not in _MODE_GRAPHS:
        raise ValueError(f"mode must be 'intersection' or 'expression', got {mode!r}")

    graph = key_added if key_added is not None else _MODE_GRAPHS.get(mode, "intr")

    adata = adata_sc.copy()

    # Extract expression connectivities (distances are not used). scanpy stores its default
    # graph as obsp['connectivities'] and a key_added=<k> one as obsp['<k>_connectivities'];
    # a custom key never falls back to the default graph.
    nb = adata.uns.get(neighbors_key)
    nb = nb if isinstance(nb, Mapping) else {}
    cands = [f"{neighbors_key}_connectivities"]
    if nb.get("connectivities_key"):
        cands.append(nb["connectivities_key"])
    if neighbors_key == "neighbors":
        cands.append("connectivities")
    conn_key = next((k for k in cands if k in adata.obsp), None)
    if conn_key is None:
        raise ValueError(
            f"No kNN graph under neighbors_key={neighbors_key!r} (checked obsp {cands}). "
            "Run sc.pp.neighbors(adata"
            + ("" if neighbors_key == "neighbors" else f", key_added={neighbors_key!r}")
            + ") first."
        )
    expr_conn = adata.obsp[conn_key].copy()
    n_neighbors = nb.get("params", {}).get("n_neighbors")

    expr_conn = sp.csr_matrix(expr_conn)
    n_cells = expr_conn.shape[0]

    _log.detail(f"Expression graph: {n_cells} cells, {expr_conn.nnz} connectivity edges")

    if adata_dist is None:
        # No spatial input: the expression graph alone -- exactly what mode="expression"
        # at alpha=0 builds from any distance matrix, since that blend never reads it.
        if mode != "expression":
            raise ValueError(
                f"mode={mode!r} needs adata_dist: the intersection graph keeps only the "
                "expression edges between spatially close cells. Without a distance "
                "matrix only mode='expression' (the plain expression graph) can be built."
            )
        expr_conn_max = expr_conn.data.max()
        int_conn_csr = expr_conn.copy()
        int_conn_csr.data = int_conn_csr.data / expr_conn_max
        # float64 after the rescale, as the blend leaves it: bit-identical to alpha=0
        int_conn_csr = int_conn_csr.astype(np.float64).tocsr()
        int_conn_csr.eliminate_zeros()
        info = {"alpha": 0.0, "p_keep": None, "expr_conn_max": float(expr_conn_max),
                "spatial": False}
        _log.detail("No adata_dist: storing the expression graph alone (alpha=0)")
    else:
        int_conn_csr, info = _blend_with_distances(
            adata, adata_dist, expr_conn, alpha, p_keep, mode)

    # Transport cost = clip(-ln(conn), floor, cap) on the union-symmetrized connectivity.
    conn_sym = int_conn_csr.maximum(int_conn_csr.T).tocsr()
    conn_sym.eliminate_zeros()
    if conn_sym.nnz == 0:
        raise ValueError(
            f"Integration left no edges at alpha={alpha}, p_keep={p_keep}, mode={mode!r}. "
            "At alpha=1.0 the expression side is dropped entirely and the blended "
            "connectivity is the prior's similarity alone, which is 0 on its longest kept "
            "edge -- so a prior with a single distinct distance yields an empty graph. "
            "Lower alpha, or raise p_keep."
        )
    raw_cost = -np.log(conn_sym.data)
    cost_cap = float(np.quantile(raw_cost, _COST_CAP_QUANTILE))
    _pos = raw_cost[raw_cost > 0.0]
    cost_floor = float(0.5 * _pos.min()) if _pos.size else cost_cap * 1e-6
    int_cost_csr = conn_sym.copy()
    int_cost_csr.data = np.clip(raw_cost, cost_floor, cost_cap)


    # Disconnected-node detection (intersection only; expression cannot orphan a cell)
    if mode == "intersection":
        degree = np.diff(int_conn_csr.indptr)
        disconnected_mask = degree == 0
        disconnected_indices = np.where(disconnected_mask)[0]
        connected_indices = np.where(~disconnected_mask)[0]
        disconnected_names = adata.obs_names[disconnected_indices].tolist()

        _log.detail(f"Connected nodes: {len(connected_indices)} / {n_cells}")
        _log.detail(
            f"Disconnected nodes: {len(disconnected_indices)} / {n_cells} "
            f"({100 * len(disconnected_indices) / n_cells:.1f}%)"
        )

    # params dict
    neighbors_graph = f"neighbors_{graph}"
    params = {
        "n_neighbors": None if n_neighbors is None else int(n_neighbors),
        "method": graph,
        **info,
        "cost_cap": cost_cap,
        "cost_floor": cost_floor,
        "cost_cap_quantile": _COST_CAP_QUANTILE,
    }

    adata.obsp[f"{neighbors_graph}_distances"] = int_cost_csr
    adata.obsp[f"{neighbors_graph}_connectivities"] = int_conn_csr
    adata.uns[neighbors_graph] = {
        "connectivities_key": f"{neighbors_graph}_connectivities",
        "distances_key": f"{neighbors_graph}_distances",
        "params": params,
    }

    if mode == "intersection":
        adata.uns[f"disconnected_{graph}"] = {
            "indices": disconnected_indices,
            "names": disconnected_names,
            "mask": disconnected_mask,
            "n_disconnected": len(disconnected_indices),
            "n_connected": len(connected_indices),
        }
        adata.obs[f"disconnected_{graph}"] = disconnected_mask

    _log.detail(f"Stored graph under key '{neighbors_graph}'")

    return adata


# ---------------------------------------------------------------------------
# pp.graphs
# ---------------------------------------------------------------------------
@_log.container()
def graphs(
    adata_sc,
    adata_dist=None,
    *,
    alpha: float = 0.3,
    p_keep: float = 0.32,
    modes=None,
    neighbors_key: str = "neighbors",
    verbose: int | bool = 1,
) -> ad.AnnData:
    """Build every requested graph mode into ONE object.

    Runs :func:`integrate` once per mode and merges the results, so e.g. a baseline
    and a spatially constrained graph sit side by side in one AnnData instead of two
    that can drift apart. With more than one graph present, the stage functions
    (``tl.endpoints``, ``tl.qrot``, ``pl.endpoints``) need ``graph=``; ``tl.trajectory``
    takes ``"intr"`` when it is stored, else the only graph there is.

    Parameters
    ----------
    adata_sc, adata_dist : anndata.AnnData
        As for :func:`integrate`. Without ``adata_dist`` only the plain expression graph
        (``"expr"``) is built.
    alpha, p_keep, neighbors_key
        Passed through to :func:`integrate` for every mode (defaults ``alpha=0.3``,
        ``p_keep=0.32``).
    modes : sequence of str, optional
        Which graphs to build, stored as ``graph="expr"`` and ``graph="intr"``. Default:
        both, or ``("expression",)`` when there is no ``adata_dist``.
    verbose : int or bool, default 1
        Logging verbosity.

    Returns
    -------
    anndata.AnnData
        One object carrying every requested graph, under ``neighbors_<graph>``.
    """
    if modes is None:
        modes = ("expression", "intersection") if adata_dist is not None else ("expression",)
    modes = list(modes)
    if not modes:
        raise ValueError("graphs(): `modes` must name at least one graph mode.")
    unknown = [m for m in modes if m not in _MODE_GRAPHS]
    if unknown:
        raise ValueError(
            f"graphs(): unknown mode(s) {unknown}; choose from {sorted(_MODE_GRAPHS)}."
        )

    out = None
    for mode in modes:
        res = integrate(
            adata_sc, adata_dist, alpha=alpha, p_keep=p_keep, mode=mode,
            neighbors_key=neighbors_key, verbose=verbose,
        )
        if out is None:
            out = res
            continue
        graph = _MODE_GRAPHS[mode]
        nk = f"neighbors_{graph}"
        out.obsp[f"{nk}_distances"] = res.obsp[f"{nk}_distances"]
        out.obsp[f"{nk}_connectivities"] = res.obsp[f"{nk}_connectivities"]
        out.uns[nk] = res.uns[nk]
        disc = f"disconnected_{graph}"        # intersection-family bookkeeping
        if disc in res.obs:
            out.obs[disc] = res.obs[disc].to_numpy()
        if disc in res.uns:
            out.uns[disc] = res.uns[disc]
    _log.detail(f"graphs {[_MODE_GRAPHS[m] for m in modes]} built into one object")
    return out
