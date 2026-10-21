"""pp: spatial distances (wasserstein), integrated graphs (integrate)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from itertools import pairwise

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from . import _log
from ._utils import _MODE_GRAPHS, _PKG_DIR, _is_positional_index


# ---------------------------------------------------------------------------
# pp.wasserstein  (consumes a cell-to-spot mapping matrix, e.g. from Tangram)
# ---------------------------------------------------------------------------
def _emd_rows(rows, nbrs_arr, nbr_lens, weights_arr, coords):
    """EMDs from each cell in rows to every later cell (condensed order, nan if undefined)."""
    import ot

    n_cells = len(nbr_lens)
    out = []
    n_dropped = n_unconverged = 0
    with warnings.catch_warnings():
        # convergence is read from POT's result code, so its per-pair warning is noise
        warnings.filterwarnings("ignore", "numItermax reached", UserWarning)
        for i in rows:
            d = np.full(n_cells - i - 1, np.nan)
            out.append(d)
            idx_i, w_i = nbrs_arr[i, :nbr_lens[i]], weights_arr[i, :nbr_lens[i]]
            if w_i.size == 0:
                continue
            for j in range(i + 1, n_cells):
                k_j = nbr_lens[j]
                if k_j == 0:
                    continue
                cost = cdist(coords[idx_i], coords[nbrs_arr[j, :k_j]])
                emd, log = ot.emd2(w_i, weights_arr[j, :k_j], cost, log=True)
                # the network simplex can land a hair below 0 for near-identical distributions
                emd = np.maximum(emd, 0)
                if not np.isfinite(emd) or log.get("result_code") == 2:
                    n_dropped += 1
                # POT's MAX_ITER_REACHED: an early-stopped plan can miss the marginals, bounding nothing
                elif log.get("result_code") == 3:
                    n_unconverged += 1
                else:
                    d[j - i - 1] = emd
    return np.concatenate(out), n_dropped, n_unconverged


@_log.stage("computing cell-cell Wasserstein distances")
def wasserstein(
    M,
    adata_sp: ad.AnnData,
    pxl_to_um: float = 1.0,
    mass_threshold: float = 0.9,
    verbose: int | bool = 1,
    n_jobs: int = -1,
) -> ad.AnnData:
    """Compute pairwise Wasserstein-1 distances between single-cell spot-mapping distributions.

    Treats each cell as a probability distribution over spatial spots (e.g. from
    Tangram or Cell2location deconvolution), using Euclidean distance between
    spatial spots as ground metric:

        W1(a, b) = min_{P >= 0}  <D, P>
        s.t.                     P 1 = a,  P^T 1 = b

    Only the top `mass_threshold` fraction of each cell's spot mass is retained
    and renormalized to keep all-pairs EMD computation tractable. Cells with identical
    distributions have distance 0.

    Parameters
    ----------
    M : anndata.AnnData or str
        Cell-to-spot mapping of shape ``(n_cells, n_spots)``, or path to an ``.h5ad``
        file. Each row is a distribution over spots. Columns are matched to
        `adata_sp` by spot name or positional index.
    adata_sp : anndata.AnnData
        Spatial reference object with spot coordinates in ``obsm["spatial"]`` of shape
        ``(n_spots, 2)``.
    pxl_to_um : float, default 1.0
        Conversion factor scaling spatial coordinates to microns.
    mass_threshold : float, default 0.9
        Per-cell cumulative mass threshold; only spots within this top fraction
        are kept for EMD calculation.
    verbose : int or bool, default 1
        Verbosity level (0: silent, 1: stages, 2: detailed progress).
    n_jobs : int, default -1
        Joblib parallel workers. ``-1`` uses all available CPU cores.

    Returns
    -------
    anndata.AnnData
        AnnData indexed by cell names where ``uns["distances"]`` contains the
        condensed pairwise distance vector (SciPy ``pdist`` order).
        Can be passed directly as `adata_dist` to :func:`deltatrace.pp.integrate`.

    Examples
    --------
    >>> import deltatrace as dt
    >>> # M: AnnData (cells x spots) from Tangram / Cell2location
    >>> # adata_sp: AnnData (spots) with obsm["spatial"]
    >>> adata_dist = dt.pp.wasserstein(M, adata_sp, pxl_to_um=1.0)  # doctest: +SKIP
    """
    import scanpy as sc
    from joblib import Parallel, delayed, effective_n_jobs

    if isinstance(M, str):
        M = sc.read_h5ad(M)

    # --- validate the mapping-side inputs ---------------------------------------
    if "spatial" not in adata_sp.obsm:
        raise KeyError(
            "adata_sp.obsm['spatial'] not found: set it to the spot coordinates, "
            "shape (n_spots, 2)."
        )
    n_spots = adata_sp.n_obs
    if M.shape[1] != n_spots:
        raise ValueError(
            f"M has {M.shape[1]} spot columns but adata_sp has {n_spots} spots; "
            "M.X must be (n_cells, n_spots)."
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
                "M.var_names or adata_sp.obs_names repeats a spot name: make the names "
                "unique."
            )
        if set(m_spots) != set(sp_spots):
            raise ValueError(
                "M.var_names and adata_sp.obs_names name different spots, e.g. "
                f"{sorted(set(m_spots) - set(sp_spots))[:3]} vs "
                f"{sorted(set(sp_spots) - set(m_spots))[:3]}. Pass the spatial object the "
                "mapping was made on, or give M a default 0, 1, 2, ... var index to match "
                "by position."
            )
        coords = coords[pd.Index(sp_spots).get_indexer(m_spots)]
        _log.detail("Matched M's spot columns to adata_sp's spots by name")
    coords = np.asarray(coords * pxl_to_um, dtype=np.float64)

    X = M.X
    if sp.issparse(X):
        X = X.tocsr()

    # Build per-cell spot lists filtered to top mass_threshold fraction
    nbrs = []
    weights = []

    for i in range(X.shape[0]):
        if sp.issparse(X):
            row = slice(X.indptr[i], X.indptr[i + 1])
            indices = X.indices[row]
            w = X.data[row]
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
    nbr_lens = np.array([len(x) for x in nbrs], dtype=np.int64)
    if n_cells and np.median(nbr_lens) > n_spots / 2:
        warnings.warn(
            "Mapping nearly uniform; consider a different mapping (a better-matched dataset "
            "or algorithm).",
            UserWarning,
            skip_file_prefixes=(_PKG_DIR,),
        )

    if _log.wants_detail():
        _log.detail(
            f"{mass_threshold * 100:.0f}% mass-threshold spots per cell: "
            f"min {np.min(nbr_lens)}, mean {np.mean(nbr_lens):.1f}, "
            f"median {np.median(nbr_lens):.0f}, max {np.max(nbr_lens)}"
        )
    _log.detail(f"{n_cells} cells, {total_pairs} pairs (n_jobs={n_jobs})")

    # padded per-cell spot arrays, which joblib memmaps for the workers
    max_k = int(nbr_lens.max()) if n_cells > 0 else 0
    nbrs_arr = np.zeros((n_cells, max_k), dtype=np.int64)
    weights_arr = np.zeros((n_cells, max_k), dtype=np.float64)
    for i, (idx, w) in enumerate(zip(nbrs, weights, strict=True)):
        nbrs_arr[i, : len(idx)] = idx
        weights_arr[i, : len(w)] = w

    # blocks of consecutive rows holding about the same number of pairs
    pairs_before = np.r_[0, np.cumsum(np.arange(n_cells - 1, 0, -1))]
    n_blocks = 8 * effective_n_jobs(n_jobs)
    cuts = np.unique(np.searchsorted(pairs_before, np.linspace(0, total_pairs, n_blocks + 1)))
    blocks = [range(a, b) for a, b in pairwise(cuts)]

    # fill one preallocated condensed vector as the blocks finish
    distances = np.empty(total_pairs)
    n_dropped = n_unconverged = 0
    gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(_emd_rows)(rows, nbrs_arr, nbr_lens, weights_arr, coords) for rows in blocks
    )
    for rows, (d, rdrop, runconv) in zip(blocks, gen, strict=True):
        distances[pairs_before[rows[0]]:pairs_before[rows[0]] + d.size] = d
        n_dropped += rdrop
        n_unconverged += runconv
        _log.detail(f"processed cells up to {rows[-1] + 1}/{n_cells}")

    notes = []
    if n_dropped:
        notes.append(
            f"{n_dropped} of {total_pairs} cell pairs had a non-finite EMD (often an "
            "empty spot distribution) and get no edge."
        )
    if n_unconverged:
        notes.append(
            f"{n_unconverged} of {total_pairs} cell pairs hit the EMD iteration cap and were "
            f"left undefined (NaN)."
        )
    if notes:
        warnings.warn(" ".join(notes), RuntimeWarning, skip_file_prefixes=(_PKG_DIR,))

    if _log.wants_detail():
        valid = distances[np.isfinite(distances)]
        _log.detail(
            f"valid pairs: {valid.size}/{total_pairs} "
            f"({100 * valid.size / total_pairs:.2f}%)"
        )
        if valid.size > 0:
            _log.detail(
                f"distance (microns): min {valid.min():.4g}, mean {valid.mean():.4g}, "
                f"median {np.median(valid):.4g}, max {valid.max():.4g}"
            )

    return ad.AnnData(
        obs=pd.DataFrame(index=M.obs_names),
        uns={
            "distances": distances,
            "d_max": float(np.fmax.reduce(distances, initial=0.0)),
            "mass_threshold": mass_threshold,
            "cost": "euclidean",
        },
    )


# Per-dataset quantile of -ln(conn) clipping transport costs. No public knob: retune here.
_COST_CAP_QUANTILE = 0.99


def _align_distance_matrix(names_sc, names_dist):
    """Each adata_sc cell's position in adata_dist by name, or None to read in order."""
    sc_names = [str(nm) for nm in names_sc]
    d_names = [str(nm) for nm in names_dist]
    if sc_names == d_names:
        return None
    if _is_positional_index(d_names):
        if not _is_positional_index(sc_names):
            warnings.warn(
                "adata_dist has no cell names, so its rows are read in adata's cell "
                "order. Give it adata's obs_names to match the cells by name.",
                UserWarning,
                skip_file_prefixes=(_PKG_DIR,),
            )
        return None
    if _is_positional_index(sc_names) and set(sc_names) != set(d_names):
        return None

    for who, names in (("adata_dist", d_names), ("adata_sc", sc_names)):
        idx = pd.Index(names)
        dupes = sorted(idx[idx.duplicated()].unique())[:3]
        if dupes:
            raise ValueError(
                f"{who}.obs_names repeats {dupes}: make the names unique "
                "(AnnData.obs_names_make_unique)."
            )

    d_idx = pd.Index(d_names)
    perm = d_idx.get_indexer(sc_names)
    if (perm < 0).any():
        missing = pd.Index(sc_names)[perm < 0].tolist()
        raise ValueError(
            f"adata_sc and adata_dist name different cells: {len(missing)} of adata_sc's "
            f"are missing from adata_dist, e.g. {missing[:3]}. If they are the same cells "
            "in the same order, give adata_dist a default 0, 1, 2, ... index."
        )

    n_moved = int((perm != np.arange(perm.size)).sum())
    _log.detail(
        f"Realigned adata_dist to adata_sc.obs_names ({n_moved} of {perm.size} rows moved)"
    )
    return perm


def _square_to_condensed(D):
    """Condensed form of a square distance matrix (NaN where a sparse one stores nothing), checked and max-symmetrized a block of rows at a time."""
    n = D.shape[0]
    if sp.issparse(D):
        D = D.tocoo(copy=True)
        D.sum_duplicates()
        rows, cols = D.row.astype(np.int64), D.col.astype(np.int64)
        vals = D.data.astype(np.float64)
        n_nonzero = np.count_nonzero(vals)
        n_bad = np.count_nonzero(~np.isfinite(vals))
        n_neg = np.count_nonzero(vals < 0)
        diag = rows == cols
        n_diag = int((vals[diag] > 0).sum())

        if not n_nonzero:
            raise ValueError("Distance matrix has no nonzero entries.")
        if n_bad:
            raise ValueError(
                f"Distance matrix has {n_bad} non-finite entries (NaN or inf); leave unknown "
                "pairs out of a sparse matrix instead."
            )
        if n_neg:
            raise ValueError(
                f"Distance matrix has {n_neg} negative entries; distances must be >= 0."
            )
        if n_diag:
            _log.detail(f"Dropping {n_diag} stored diagonal entries")
            if n_nonzero == n_diag:
                raise ValueError("Distance matrix has no off-diagonal nonzero entries.")

        off = ~diag
        r, c, v = rows[off], cols[off], vals[off]
        upper = r < c
        idx_u = _pair_index(n, r[upper], c[upper])
        val_u = v[upper]
        idx_l = _pair_index(n, c[~upper], r[~upper])
        val_l = v[~upper]

        all_idx = np.concatenate([idx_u, idx_l])
        uniq_idx, counts = np.unique(all_idx, return_counts=True)
        n_single = int(np.count_nonzero(counts == 1))
        both_mask_u = np.isin(idx_u, uniq_idx[counts == 2])
        both_mask_l = np.isin(idx_l, uniq_idx[counts == 2])
        idx_both_u, val_both_u = idx_u[both_mask_u], val_u[both_mask_u]
        idx_both_l, val_both_l = idx_l[both_mask_l], val_l[both_mask_l]
        o_u = np.argsort(idx_both_u)
        o_l = np.argsort(idx_both_l)
        diff_val = int(np.count_nonzero(val_both_u[o_u] != val_both_l[o_l]))
        n_asym = 2 * (n_single + diff_val)

        dc = np.full(n * (n - 1) // 2, np.nan)
        np.fmax.at(dc, idx_u, val_u)
        np.fmax.at(dc, idx_l, val_l)
        if n_asym:
            _log.detail(f"Symmetrizing distance matrix ({n_asym} asymmetric entries)")
        return dc

    dc = np.empty(n * (n - 1) // 2)
    row_start = np.r_[0, np.cumsum(np.arange(n - 1, 0, -1))]
    n_nonzero = n_bad = n_neg = n_diag = n_asym = 0
    step = max(1, 2**22 // max(n, 1))
    for s in range(0, n, step):
        B = vals = np.asarray(D[s:s + step], dtype=np.float64)
        n_nonzero += np.count_nonzero(vals)
        n_bad += np.count_nonzero(~np.isfinite(vals))
        n_neg += np.count_nonzero(vals < 0)
        for r, row in enumerate(B, start=s):
            # upper triangle in place, lower triangle merged by max (undirected prior)
            n_diag += row[r] > 0
            dc[row_start[r]:row_start[r] + n - r - 1] = row[r + 1:]
            pos = row_start[:r] + r - np.arange(r) - 1
            same = (dc[pos] == row[:r]) | (np.isnan(dc[pos]) & np.isnan(row[:r]))
            n_asym += 2 * np.count_nonzero(~same)
            dc[pos] = np.fmax(dc[pos], row[:r])

    if not n_nonzero:
        raise ValueError("Distance matrix has no nonzero entries.")
    # a non-finite entry is a bad input, not a missing distance
    if n_bad:
        raise ValueError(
            f"Distance matrix has {n_bad} non-finite entries (NaN or inf); leave unknown "
            "pairs out of a sparse matrix instead."
        )
    if n_neg:
        raise ValueError(
            f"Distance matrix has {n_neg} negative entries; distances must be >= 0."
        )
    # a cell's distance to itself is never an edge
    if n_diag:
        _log.detail(f"Dropping {n_diag} stored diagonal entries")
        if n_nonzero == n_diag:
            raise ValueError("Distance matrix has no off-diagonal nonzero entries.")
    if n_asym:
        _log.detail(f"Symmetrizing distance matrix ({n_asym} asymmetric entries)")
    return dc


def _pair_index(n, i, j):
    """Position of pair (i, j), i < j, in scipy's condensed (pdist) order."""
    return n * i - i * (i + 1) // 2 + j - i - 1


def _keep_threshold(dc, p_keep):
    """(threshold, d_max, n distances, n kept, max) of dc's finite d >= 0; the quantile counts each pair twice, as the square matrix does."""
    if not 0 <= p_keep <= 1:
        raise ValueError("Quantiles must be in the range [0, 1]")
    chunk, n_bins = 1 << 24, 1 << 16

    def distance_chunks():
        for s in range(0, dc.size, chunk):
            v = dc[s:s + chunk]
            yield v[(v >= 0) & np.isfinite(v)]

    # largest distance and a histogram of all distances (no copy of dc)
    top = max((v.max() for v in distance_chunks() if v.size), default=0.0)
    if top == 0:
        raise ValueError("Distance matrix has no nonzero entries.")

    def bin_of(v):
        return np.minimum((v / top * n_bins).astype(np.int64), n_bins - 1)

    counts = sum(np.bincount(bin_of(v), minlength=n_bins) for v in distance_chunks())
    n_dist = int(counts.sum())

    # np.quantile over the square matrix's 2 n_dist values interpolates between two order statistics
    n_square = 2 * n_dist
    position = (n_square - 1) * float(p_keep)
    below = int(position)
    ranks = np.array([below, min(below + 1, n_square - 1)]) // 2

    # read those two values exactly: sort only the histogram bins that hold them
    cum_counts = np.cumsum(counts)
    rank_bins = np.searchsorted(cum_counts, ranks, side="right")
    first_rank = cum_counts - counts
    candidates = np.sort(np.concatenate([v[np.isin(bin_of(v), rank_bins)] for v in distance_chunks()]))
    a, b = candidates[ranks - first_rank[rank_bins[0]]]
    threshold = np.quantile([a, b], position - below)

    # kept distances: at most the threshold
    n_kept, d_max = 0, 0.0
    for v in distance_chunks():
        v = v[v <= threshold]
        if v.size:
            n_kept, d_max = n_kept + v.size, max(d_max, v.max())
    return threshold, d_max, n_dist, n_kept, top


def _blend_with_distances(adata, adata_dist, expr_conn, alpha, p_keep, mode, _cached=None):
    """The spatial half of :func:`integrate`: blended connectivities and their params."""
    n_cells = expr_conn.shape[0]

    if _cached is not None and "dc" in _cached:
        threshold = _cached["threshold"]
        d_max = _cached["d_max"]
        d_top = _cached["d_top"]
        n_kept = _cached["n_kept"]
        expr_conn_max = _cached["expr_conn_max"]
        d = _cached["d"]
        keep = _cached["keep"]
        conn = _cached["conn"]
    else:
        # condensed vector, or a square matrix to condense once the names are checked
        if isinstance(adata_dist, pd.DataFrame):
            if not adata_dist.index.equals(adata_dist.columns):
                raise ValueError(
                    "adata_dist's row and column labels must name the same cells in the "
                    "same order."
                )
            names, D = adata_dist.index, adata_dist.to_numpy()
        elif "distances" in adata_dist.uns:
            names, D = adata_dist.obs_names, np.asarray(adata_dist.uns["distances"])
        else:
            names, D = adata_dist.obs_names, adata_dist.X
        n_dist = len(names) if D.ndim == 1 else D.shape[0]
        if n_dist != n_cells:
            raise ValueError(f"Cell count mismatch: sc={n_cells}, dist={n_dist}")
        if D.ndim == 1 and D.size != n_cells * (n_cells - 1) // 2:
            raise ValueError(
                f"adata_dist.uns['distances'] holds {D.size} distances; {n_cells} cells "
                f"have {n_cells * (n_cells - 1) // 2} pairs."
            )
        if D.ndim == 2 and D.shape[1] != n_cells:
            raise ValueError(
                f"Distance matrix must be square over the cells: got {D.shape[0]}x"
                f"{D.shape[1]} for {n_cells} cells."
            )
        perm = _align_distance_matrix(adata.obs_names, names)
        dc = D if D.ndim == 1 else _square_to_condensed(D)

        threshold, d_max, n_pos, n_kept, d_top = _keep_threshold(dc, p_keep)
        _log.detail(
            f"Distance filtering: kept {2 * n_kept} / {2 * n_pos} edges "
            f"(p_keep={p_keep}, threshold={threshold:.4g})"
        )
        expr_conn_max = expr_conn.data.max()
        _log.detail(
            f"Normalization - expr_conn_max: {expr_conn_max:.4g}, "
            f"ext_dist_max: {d_max:.4g}"
        )

        # distance of each expression edge u-v, looked up in the condensed vector
        u = np.repeat(np.arange(n_cells), np.diff(expr_conn.indptr))
        v = expr_conn.indices.astype(np.int64)
        if perm is not None:
            u, v = perm[u], perm[v]
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        off_diag = lo < hi
        d = np.full(expr_conn.nnz, np.nan)
        d[off_diag] = dc[_pair_index(n_cells, lo[off_diag], hi[off_diag])]

        # kept edges: (1 - alpha) expression connectivity + alpha spatial similarity 1 - d/d_max
        keep = (d >= 0) & (d <= threshold)
        conn = (expr_conn.data / expr_conn_max).astype(np.float64)

        if _cached is not None:
            _cached.update(
                dc=dc,
                threshold=threshold,
                d_max=d_max,
                d_top=d_top,
                n_kept=n_kept,
                expr_conn_max=expr_conn_max,
                d=d,
                keep=keep,
                conn=conn,
            )

    # when every kept distance is 0 (threshold 0), each kept pair is fully similar
    blended = (1 - alpha) * conn + alpha * (1.0 - d / (d_max or 1.0))
    not_kept = 0.0 if mode == "intersection" else (1 - alpha) * conn
    int_conn_csr = sp.csr_matrix(
        (np.where(keep, blended, not_kept), expr_conn.indices, expr_conn.indptr),
        shape=expr_conn.shape,
        copy=True,
    )
    int_conn_csr.eliminate_zeros()

    info = {
        "alpha": alpha,
        "p_keep": p_keep,
        "expr_conn_max": float(expr_conn_max),
        "ext_dist_max": float(d_max),
    }
    if mode == "expression":
        n_both = int(keep.sum())
        n_expr_only = int(np.count_nonzero(conn[~keep]))
        _log.detail(f"Integration: {n_both} edges in both, {n_expr_only} expr-only")
        _log.detail(f"Integrated graph: {int_conn_csr.nnz} total edges")
        info.update(n_edges_both=n_both, n_edges_expr_only=n_expr_only)
    else:
        _log.detail(
            f"Intersection: {int(keep.sum())} edges "
            f"(from expr={expr_conn.nnz}, dist={2 * n_kept})"
        )
        info.update(
            ext_dist_threshold_raw=float(threshold),
            ext_dist_unfiltered_max=float(d_top),
            n_edges_intersection=int(int_conn_csr.nnz),
            n_edges_expr=int(expr_conn.nnz),
            n_edges_dist_filtered=2 * n_kept,
        )
    return int_conn_csr, info


# ---------------------------------------------------------------------------
# pp.integrate
# ---------------------------------------------------------------------------
def _integrate_message(a) -> str:
    graph = a.get("key_added") or _MODE_GRAPHS.get(a.get("mode"), a.get("mode"))
    return (f"building the '{graph}' graph"
            + (" from expression alone" if a.get("adata_dist") is None else ""))


@_log.stage(_integrate_message)
def _integrate_single(
    adata_sc,
    adata_dist=None,
    alpha: float = 0.3,
    p_keep: float = 0.3,
    mode: str = "intersection",
    neighbors_key: str = "neighbors",
    key_added: str | None = None,
    verbose: int | bool = 1,
    *,
    _dist_cache=None,
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
    adata_dist : anndata.AnnData, pandas.DataFrame or str, optional
        Cell-cell distances: an AnnData from :func:`wasserstein` (condensed vector in
        ``uns["distances"]``, NaN for undefined pairs), or a square ``(n_cells, n_cells)``
        matrix as an AnnData's ``X`` (dense or sparse) or a DataFrame labeled by cell on
        both axes; a path to an ``.h5ad`` is also accepted. A square matrix is
        max-symmetrized and its diagonal ignored. Every entry of a dense matrix is a
        distance, 0 included; a sparse matrix's absent entries are no distance, and its
        stored zeros are distance 0. Aligned to ``adata_sc`` by cell name; without cell
        names it is read in ``adata_sc``'s order (with a warning when ``adata_sc`` has
        cell names). Integration works on the condensed vector (``4 n_cells^2`` bytes); a
        square input is converted to it a block of rows at a time. ``None`` (no spatial
        input) builds the plain expression graph: ``mode="expression"`` only, and
        ``alpha``/``p_keep`` are ignored.
    alpha : float, default 0.3
        Spatial blending weight in ``[0, 1]``. ``0`` drops the spatial term: with
        ``mode="expression"`` the result is then the plain expression graph.
    p_keep : float, default 0.3
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
                f"mode={mode!r} needs adata_dist; without it only mode='expression' works."
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
            adata, adata_dist, expr_conn, alpha, p_keep, mode, _cached=_dist_cache)

    # Transport cost = clip(-ln(conn), floor, cap) on the union-symmetrized connectivity.
    conn_sym = int_conn_csr.maximum(int_conn_csr.T).tocsr()
    conn_sym.eliminate_zeros()
    if conn_sym.nnz == 0:
        raise ValueError(
            f"Integration left no edges at alpha={alpha}, p_keep={p_keep}, mode={mode!r}: "
            "lower alpha or raise p_keep."
        )
    raw_cost = -np.log(conn_sym.data)
    cost_cap = float(np.quantile(raw_cost, _COST_CAP_QUANTILE))
    _pos = raw_cost[raw_cost > 0.0]
    cost_floor = float(0.5 * _pos.min()) if _pos.size else cost_cap * 1e-6
    int_cost_csr = conn_sym.copy()
    int_cost_csr.data = np.clip(raw_cost, cost_floor, cost_cap)


    # Disconnected-node detection (intersection only; expression cannot orphan a cell)
    if mode == "intersection":
        degree = np.diff(conn_sym.indptr)
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
# pp.integrate
# ---------------------------------------------------------------------------
def integrate(
    adata_sc,
    adata_dist=None,
    alpha: float = 0.3,
    p_keep: float = 0.3,
    mode: str | None = None,
    neighbors_key: str = "neighbors",
    key_added: str | None = None,
    verbose: int | bool = 1,
    *,
    modes: Sequence[str] | None = None,
) -> ad.AnnData:
    """Integrate a single-cell expression kNN graph with spatial cell-cell distances.

    Combines transcriptomic similarities and physical/Wasserstein distances:

        conn = (1 - alpha) * expr_conn + alpha * spatial_sim
        cost = clip(-ln(conn), floor, cap)

    In intersection mode (``"intr"``), only edges satisfying the spatial distance
    quantile threshold (`p_keep`) are kept. In expression mode (``"expr"``), all
    expression edges are retained with spatial penalties.

    By default, when `adata_dist` is provided, both ``"expr"`` and ``"intr"``
    graphs are computed and stored into `adata_sc`. When `adata_dist=None`, only
    the normalized expression graph (``"expr"``) is built.

    Parameters
    ----------
    adata_sc : anndata.AnnData or str
        Single-cell AnnData with a kNN graph in ``.obsp``, or path to an ``.h5ad`` file.
    adata_dist : anndata.AnnData, pandas.DataFrame or str, optional
        Cell-cell spatial distance output from :func:`wasserstein` (condensed vector
        in ``uns["distances"]``) or square distance matrix. If None, builds the
        expression graph alone.
    alpha : float, default 0.3
        Spatial blending weight in ``[0, 1]``. ``alpha=0`` uses expression only.
    p_keep : float, default 0.3
        Distance quantile cutoff for intersection filtering. Edges with distance
        above this quantile are pruned.
    mode : {"intersection", "expression"}, optional
        Graph mode to build. If None, builds both ``"expression"`` and ``"intersection"``
        when `adata_dist` is provided, or ``"expression"`` when `adata_dist` is None.
    neighbors_key : str, default "neighbors"
        Key in `adata_sc` for the base expression kNN graph.
    key_added : str, optional
        Custom key suffix to store the graph under (``neighbors_<key_added>``).
    verbose : int or bool, default 1
        Verbosity level (0: silent, 1: stages, 2: detailed progress).
    modes : sequence of str, optional
        Explicit sequence of graph modes to construct (e.g. ``("expression", "intersection")``).

    Returns
    -------
    anndata.AnnData
        Updated AnnData object carrying the integrated graph(s) under
        ``obsp["neighbors_<graph>_connectivities"]`` and
        ``obsp["neighbors_<graph>_distances"]``.

    Examples
    --------
    Spatial integration combining expression and spatial distances:

    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist, alpha=0.3, p_keep=0.3)
    >>> "neighbors_intr" in adata.uns and "neighbors_expr" in adata.uns
    True

    Expression-only baseline (no spatial data):

    >>> adata = dt.pp.integrate(adata, mode="expression")
    """
    if key_added is not None:
        return _integrate_single(
            adata_sc,
            adata_dist,
            alpha=alpha,
            p_keep=p_keep,
            mode=mode or "intersection",
            neighbors_key=neighbors_key,
            key_added=key_added,
            verbose=verbose,
        )

    if modes is not None:
        modes = list(modes)
    elif mode == "intersection":
        if adata_dist is None:
            raise ValueError("integrate(): mode='intersection' needs adata_dist.")
        modes = ["expression", "intersection"]
    elif mode == "expression":
        modes = ["expression"]
    elif mode is not None:
        raise ValueError(f"mode must be 'intersection' or 'expression', got {mode!r}")
    else:
        modes = ["expression", "intersection"] if adata_dist is not None else ["expression"]

    if not modes:
        raise ValueError("integrate(): `modes` must name at least one graph mode.")
    unknown = [m for m in modes if m not in _MODE_GRAPHS]
    if unknown:
        raise ValueError(
            f"integrate(): unknown mode(s) {unknown}; choose from {sorted(_MODE_GRAPHS)}."
        )

    _dist_cache = {} if adata_dist is not None else None
    out = None
    for m in modes:
        res = _integrate_single(
            adata_sc,
            adata_dist,
            alpha=alpha,
            p_keep=p_keep,
            mode=m,
            neighbors_key=neighbors_key,
            verbose=verbose,
            _dist_cache=_dist_cache,
        )
        if out is None:
            out = res
            continue
        graph = _MODE_GRAPHS[m]
        nk = f"neighbors_{graph}"
        out.obsp[f"{nk}_distances"] = res.obsp[f"{nk}_distances"]
        out.obsp[f"{nk}_connectivities"] = res.obsp[f"{nk}_connectivities"]
        out.uns[nk] = res.uns[nk]
        disc = f"disconnected_{graph}"
        if disc in res.obs:
            out.obs[disc] = res.obs[disc].to_numpy()
        if disc in res.uns:
            out.uns[disc] = res.uns[disc]

    _log.detail(f"integrated graph(s) {[_MODE_GRAPHS[m] for m in modes]} built into one object")
    return out
