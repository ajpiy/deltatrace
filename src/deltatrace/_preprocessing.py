"""pp: spatial distances (wasserstein), integrated graphs (integrate, graphs)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from itertools import pairwise

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
                if not np.isfinite(emd):
                    n_dropped += 1
                # POT's MAX_ITER_REACHED: an early-stopped plan can miss the marginals, bounding nothing
                elif log["result_code"] == 3:
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
    """Earth-Mover's distance between cells' spot-mapping distributions.

    Each cell is a distribution over spatial spots (the cell-to-spot mapping ``M``,
    e.g. from tangram); spot-spot Euclidean distance D is the ground metric:

        W1(a, b) = min_{P >= 0}  <D, P>
        s.t.                     P 1 = a,  P^T 1 = b

    Only the top ``mass_threshold`` fraction of each cell's mass is kept, which is
    what keeps the all-pairs cost tractable. Cells with the same distribution are at
    distance 0, which :func:`integrate` keeps as the closest pair. Pairs whose EMD comes
    back non-finite, or that hit the solver's iteration cap, are left undefined (NaN) and
    counted in one ``RuntimeWarning``. A mapping whose median cell keeps over half of all
    spots is warned about as nearly uniform.

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
        Joblib workers; ``-1`` uses all cores, ``1`` runs in this process. Every cell
        pair is solved, so this is the step that grows fastest with the number of cells.
        The result does not depend on it. Starting the workers costs about a second,
        and joblib falls back to one process where it cannot spawn any.

    Returns
    -------
    anndata.AnnData
        Indexed by ``M``'s cell names, with no variables;
        ``uns["distances"]`` holds every pairwise distance (microns) once, as a
        float64 vector in SciPy's condensed ``pdist`` order
        (``scipy.spatial.distance.squareform`` gives the square matrix). That is
        ``4 n_cells^2`` bytes, half a dense matrix. :func:`integrate` /
        :func:`graphs` read it as ``adata_dist``.
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
    D = D.tocsr() if sp.issparse(D) else D
    dc = np.empty(n * (n - 1) // 2)
    row_start = np.r_[0, np.cumsum(np.arange(n - 1, 0, -1))]
    n_nonzero = n_bad = n_neg = n_diag = n_asym = 0
    step = max(1, 2**22 // max(n, 1))
    for s in range(0, n, step):
        B = D[s:s + step]
        if sp.issparse(B):
            # a stored entry is a distance (0 included), an absent one is not
            B = B.tocoo()
            B.sum_duplicates()
            vals = B.data.astype(np.float64)
            rows, cols, B = B.row, B.col, np.full(B.shape, np.nan)
            B[rows, cols] = vals
        else:
            B = vals = np.asarray(B, dtype=np.float64)
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


def _blend_with_distances(adata, adata_dist, expr_conn, alpha, p_keep, mode):
    """The spatial half of :func:`integrate`: blended connectivities and their params."""
    n_cells = expr_conn.shape[0]

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


@_renamed_kwargs(tag="key_added")
@_log.stage(_integrate_message)
def integrate(
    adata_sc,
    adata_dist=None,
    alpha: float = 0.3,
    p_keep: float = 0.3,
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
            adata, adata_dist, expr_conn, alpha, p_keep, mode)

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
    p_keep: float = 0.3,
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
    adata_sc, adata_dist
        As for :func:`integrate`. Without ``adata_dist`` only the plain expression graph
        (``"expr"``) is built.
    alpha, p_keep, neighbors_key
        Passed through to :func:`integrate` for every mode (defaults ``alpha=0.3``,
        ``p_keep=0.3``).
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
