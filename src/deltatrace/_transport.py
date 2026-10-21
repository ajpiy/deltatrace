"""tl: the transport solve (qrot), the velocity grid, and trajectory() composing the stages."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spla
from numba import njit
from scipy.optimize import linprog

from . import _log
from ._endpoints import endpoints
from ._lineages import _check_terminal_cut, lineages, suggest_omega, terminality
from ._utils import (
    _PKG_DIR,
    CellGraph,
    _as_labels,
    _check_labels,
    _graph_distances_key,
    _off_lcc_mask,
    _resolve_basis,
    _resolve_graph,
    _suffix,
)


# ---------------------------------------------------------------------------
# tl.qrot  (+ pseudotime reconnection helper)
# ---------------------------------------------------------------------------
def _reconnect_pseudotime(
    adata,
    pt_dict: dict,
    graph: str = "intr",
    reconnect_graph: str = "expr",
) -> dict:
    """Give every off-LCC cell the pseudotime of its nearest connected cell, by
    Dijkstra on the reconnect_graph graph (unreachable cells fall back to the LCC
    median). Returns the full {cell name: pseudotime} mapping."""
    # Find disconnected / off-LCC cells
    disc_mask = _off_lcc_mask(adata, graph)

    if disc_mask is None:
        disc_col = f"disconnected_{graph}"
        fallback_col = "disconnected_intr"
        if disc_col in adata.obs.columns:
            disc_mask = adata.obs[disc_col].astype(bool).values
        elif fallback_col in adata.obs.columns:
            disc_mask = adata.obs[fallback_col].astype(bool).values
        else:
            _log.detail("No disconnected nodes: all pseudotimes already assigned.")
            return dict(pt_dict)

    disc_names = list(adata.obs_names[disc_mask])

    if len(disc_names) == 0:
        _log.detail("No off-LCC nodes: all pseudotimes already assigned.")
        return dict(pt_dict)

    clust_names = adata.obs_names
    name_to_idx = {name: i for i, name in enumerate(clust_names)}

    # Identify connected donor cells in LCC with known pseudotime
    donor = ~clust_names.isin(disc_names) & clust_names.isin(list(pt_dict))
    connected_names_list = list(clust_names[donor])
    connected_indices = np.flatnonzero(donor)

    # Load distance matrix for reconnection graph
    dist_key = _graph_distances_key(adata, reconnect_graph)
    if dist_key is None:
        raise ValueError(
            f"No graph {reconnect_graph!r} to reconnect off-LCC cells through. Pass a graph "
            "from pp.integrate (e.g. reconnect_graph='expr') or a scanpy key_added."
        )
    D_clust = adata.obsp[dist_key].tocsr()

    # Max-symmetrize distance matrix and clear diagonal
    dist_sym = (D_clust + D_clust.T) / 2
    if dist_sym.diagonal().any():    # setdiag on an empty diagonal inserts zeros (scipy<1.13 warns)
        dist_sym.setdiag(0)
    dist_sym.eliminate_zeros()

    _targets = [n for n in sorted(disc_names) if n in name_to_idx]
    if _targets and not connected_indices.size:
        raise ValueError(
            "No LCC cell of adata has a pseudotime: no donor to reconnect the "
            f"{len(_targets)} off-LCC cells from."
        )
    _log.detail(f"Reconnecting {len(_targets)} off-LCC cells from "
                f"{len(connected_indices)} LCC donors...")
    # Dijkstra search from all donors to find nearest donor distance per cell
    d_nearest = csg.dijkstra(dist_sym, indices=connected_indices, min_only=True)

    pseudotime_full = dict(pt_dict)
    assigned = 0
    fallback = []

    # Reassign pseudotime from closest reachable donor
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

    # Fall back to median LCC pseudotime for completely unreachable cells
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
        n_total = len(adata.obs_names)
        _log.detail(
            f"pseudotime reassignment: {len(connected_names_list)} connected, "
            f"{assigned} reassigned via geodesic"
            + (f", {len(fallback)} median fallback" if fallback else "")
            + f" ({len(pseudotime_full)}/{n_total} total)"
        )

    return pseudotime_full


def _median_off_lcc_pseudotime(adata, pt_dict: dict, graph: str | None = None) -> dict:
    """Set every off-LCC cell's pseudotime to the median of the LCC cells' (an
    explicit, reproducible placeholder, not an inferred position). No-op if
    adata is None or there are no off-LCC cells."""
    pt_full = dict(pt_dict)
    if adata is None:
        return pt_full

    # Locate off-LCC cells
    off_mask = _off_lcc_mask(adata, graph)
    if off_mask is None:
        _log.detail("No graph to locate the LCC: off-LCC pseudotimes left as solved.")
        return pt_full

    names = np.asarray([str(nm) for nm in adata.obs_names])
    off_names = [nm for nm in names[off_mask] if nm in pt_full]
    if not off_names:
        _log.detail("No off-LCC cells: every pseudotime came from the solved LCC.")
        return pt_full

    # Compute median pseudotime of connected LCC cells
    lcc_vals = np.array([pt_full[nm] for nm in names[~off_mask] if nm in pt_full], float)
    lcc_vals = lcc_vals[np.isfinite(lcc_vals)]
    if lcc_vals.size == 0:
        _log.detail(f"No finite LCC pseudotime: {len(off_names)} off-LCC cells left as is.")
        return pt_full

    # Assign LCC median placeholder to off-LCC cells
    median_pt = float(np.median(lcc_vals))
    for nm in off_names:
        pt_full[nm] = median_pt
    _log.detail(f"reconnect=False: set {len(off_names)} off-LCC cells to the LCC median "
                f"pseudotime {median_pt:.6f} (from {lcc_vals.size} LCC cells)")
    return pt_full


# ---------------------------------------------------------------------------
# QROT solver: the dual, by a capped trust region then Newton (Armijo line search, tau continuation, midpoint of
# the free potentials)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _armijo_step(s, dd, slope, tau):
    """Largest t in 1, 1/2, 1/4, ... (down to 1e-12) with dual(p + t d) - dual(p) >= 1e-4 t slope.

    s = p[head] - p[tail] - c and dd = d[head] - d[tail] per edge; the gain is summed per edge to
    avoid cancellation, in one pass over the edges per trial t."""
    # Initialize trial step size t
    t = 1.0
    while t > 1e-12:
        acc = 0.0
        # Sum exact quadratic dual gain across edges to avoid cancellation
        for e in range(s.shape[0]):
            ds = t * dd[e]
            s1 = s[e] + ds
            if s[e] > 0:
                acc += -ds * ds if s1 > 0 else s[e] * (2 * ds + s[e])
            elif s1 > 0:
                acc += -s1 * s1
        # Check Armijo sufficient decrease condition
        if t * slope + acc / (2 * tau) >= 1e-4 * t * slope:
            return t
        # Backtrack: halve step size
        t /= 2
    return t


@njit(cache=True)
def _slack_residual(p, tail, head, c, f, tau, tol_abs):
    """Edge slacks s = p[head] - p[tail] - c, Kirchhoff residual r of J = s_+ / tau, and whether |r| is within
    tol_abs plus the rounding of the potentials, in one pass over the edges (same sums, in the same order, as
    the bincounts they replace)."""
    n, m = f.shape[0], head.shape[0]
    s = np.empty(m)
    inflow, outflow = np.zeros(n), np.zeros(n)
    w_in, w_out = np.zeros(n), np.zeros(n)
    ulp = np.finfo(np.float64).eps
    # Single pass over edges: compute slacks, flows, and potential rounding bounds
    for e in range(m):
        ph, pt = p[head[e]], p[tail[e]]
        se = ph - pt - c[e]
        s[e] = se
        J = max(se, 0.0) / tau
        inflow[head[e]] += J
        outflow[tail[e]] += J
        # Track numerical precision threshold from potential magnitude
        rnd = ulp * (abs(ph) + abs(pt))
        w = rnd / tau if se > -rnd else 0.0
        w_in[head[e]] += w
        w_out[tail[e]] += w
    # Kirchhoff residual: demand minus net flow
    r = f - inflow + outflow
    # Check if residual is within absolute tolerance plus potential rounding
    bound = tol_abs + w_in + w_out
    return s, r, bool(np.all(np.abs(r) <= bound))


@njit(cache=True)
def _grounded_free(idx, lab, k):
    """The cells of idx (ascending) except the first of each piece lab, which grounds that piece."""
    seen = np.zeros(k, dtype=np.bool_)
    keep = np.ones(idx.shape[0], dtype=np.bool_)
    # Ground the first node encountered in each component piece
    for i in range(idx.shape[0]):
        if seen[lab[idx[i]]]:
            continue
        seen[lab[idx[i]]] = True
        keep[i] = False
    return idx[keep]


@njit(cache=True)
def _count_lower(indptr, indices, order, budget):
    """Nonzeros of the strict lower Cholesky factor of the symmetric pattern A[order][:, order] (CSR indptr,
    indices), counted while building the elimination tree (row subtrees, one pass); budget + 1 once the count
    exceeds budget, so a dense factor is detected without counting it."""
    n = order.shape[0]
    # Build inverse permutation mapping
    rank = np.empty(n, np.int64)
    for k in range(n):
        rank[order[k]] = k
    parent = np.full(n, -1, np.int64)
    flag = np.full(n, -1, np.int64)
    count = 0
    # Traverse elimination tree row by row
    for k in range(n):
        v = order[k]
        flag[k] = k
        for q in range(indptr[v], indptr[v + 1]):
            i = rank[indices[q]]
            if i >= k:
                continue
            # Follow path in elimination tree up to root
            while flag[i] != k:
                flag[i] = k
                count += 1
                if parent[i] == -1:
                    parent[i] = k
                i = parent[i]
            # Early exit if nonzeros exceed sparsity budget
            if count > budget:
                return budget + 1
    return count


@njit(cache=True)
def _tr_eval(p, tail, head, c, f, tau, scale):
    """Negative dual objective and its gradient (scaled by scale) in one pass over the edges."""
    n, m = f.shape[0], tail.shape[0]
    # Initialize gradient with negative demand
    g = -f.copy()
    jj = 0.0
    # Accumulate active edge flow contributions to gradient and quadratic objective
    for e in range(m):
        u, v = tail[e], head[e]
        s = p[v] - p[u] - c[e]
        if s > 0.0:
            J = s / tau
            jj += J * J
            g[v] += J
            g[u] -= J
    # Scale linear demand term and gradient entries
    fp = 0.0
    for i in range(n):
        fp += f[i] * p[i]
        g[i] /= scale
    return -(fp - 0.5 * tau * jj) / scale, g


@njit(cache=True)
def _tr_hessp(tail, head, d, inv_tau_scale, n):
    """Active-Laplacian Hessian-vector product (B_a B_a^T d) / (tau * scale)."""
    out = np.zeros(n)
    # Apply Laplacian of active edges to direction vector d
    for e in range(tail.shape[0]):
        u, v = tail[e], head[e]
        dv = (d[v] - d[u]) * inv_tau_scale
        out[v] += dv
        out[u] -= dv
    return out


@njit(cache=True)
def _trust_ncg(p, tail, head, c, f, tau, scale, eps=1e-8, gtol=1e-4, maxiter=100, max_cg=20, ftol=1e-6):
    """Steihaug-Toint Newton-CG trust-region warm start on the negative dual (Nocedal & Wright Alg. 7.2).

    Uses Newton's active edges (s > -eps) so shortest-path tree edges at s == 0 provide curvature from the start,
    TRON-style inner CG iteration capping (max_cg=20), relative dual objective plateau early exit (ftol=1e-6),
    and two-tier trust-radius expansion (3x when rho > 0.9, 2x when rho > 0.75)."""
    n = p.shape[0]
    # Initialize trust region bounds and tolerance
    trust_radius, max_trust_radius, eta = 1.0, 1000.0, 0.05
    inv_tau_scale = 1.0 / (tau * scale)
    # Evaluate initial objective and gradient
    fun, jac = _tr_eval(p, tail, head, c, f, tau, scale)
    jac_mag = np.sqrt(np.sum(jac * jac))
    k = 0
    plateau_count = 0
    # Outer trust region iteration loop
    while jac_mag >= gtol and k < maxiter:
        if not np.isfinite(jac_mag):
            break
        # Identify active edge set for Hessian-vector curvature
        s = p[head] - p[tail] - c
        a = s > -eps
        th, hd = tail[a], head[a]
        # Set inner CG stopping tolerance
        tol = min(0.5, np.sqrt(jac_mag)) * jac_mag
        z = np.zeros(n)
        r = jac.copy()
        d = -r
        r_sq = np.sum(r * r)
        hits_boundary = False
        cg_iter = 0
        step = z
        # Inner Steihaug-Toint Conjugate Gradient loop
        while cg_iter < max_cg:
            cg_iter += 1
            # Compute Hessian-vector product
            Bd = _tr_hessp(th, hd, d, inv_tau_scale, n)
            dBd = np.sum(d * Bd)
            # Handle negative or zero curvature: intercept trust region boundary
            if dBd <= 0.0 or not np.isfinite(dBd):
                da = np.sum(d * d)
                db = 2.0 * np.sum(z * d)
                dc = np.sum(z * z) - trust_radius * trust_radius
                disc = np.sqrt(max(db * db - 4.0 * da * dc, 0.0))
                aux = db + (disc if db >= 0.0 else -disc)
                tb = -2.0 * dc / aux if aux != 0.0 else 0.0
                step = z + tb * d
                hits_boundary = True
                break
            # Compute CG step length
            alpha = r_sq / dBd
            z_next = z + alpha * d
            # Check if step leaves trust region sphere
            if np.sum(z_next * z_next) >= trust_radius * trust_radius:
                da = np.sum(d * d)
                db = 2.0 * np.sum(z * d)
                dc = np.sum(z * z) - trust_radius * trust_radius
                disc = np.sqrt(max(db * db - 4.0 * da * dc, 0.0))
                aux = db + (disc if db >= 0.0 else -disc)
                tb = -2.0 * dc / aux if aux != 0.0 else 0.0
                step = z + tb * d
                hits_boundary = True
                break
            # Update residual and test convergence
            r_next = r + alpha * Bd
            r_next_sq = np.sum(r_next * r_next)
            if not np.isfinite(r_next_sq):
                step = z_next
                break
            if np.sqrt(r_next_sq) < tol:
                step = z_next
                break
            # Update conjugate direction
            d = -r_next + (r_next_sq / r_sq) * d
            z = z_next
            step = z
            r = r_next
            r_sq = r_next_sq

        # Evaluate model predicted reduction
        step_sq = np.sum(step * step)
        if not np.isfinite(step_sq):
            break
        Bstep = _tr_hessp(th, hd, step, inv_tau_scale, n)
        pred_red = -(np.sum(jac * step) + 0.5 * np.sum(step * Bstep))
        if pred_red <= 0.0 or not np.isfinite(pred_red):
            break

        # Check for objective value plateau
        rel_gain = pred_red / (abs(fun) + 1.0)
        if rel_gain < ftol:
            plateau_count += 1
            if plateau_count >= 3:
                break
        else:
            plateau_count = 0

        # Evaluate actual reduction at proposed step
        p_prop = p + step
        fun_prop, jac_prop = _tr_eval(p_prop, tail, head, c, f, tau, scale)
        if not np.isfinite(fun_prop):
            break
        rho = (fun - fun_prop) / pred_red
        if not np.isfinite(rho):
            break
        # Update trust region radius based on agreement ratio rho
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.9 and hits_boundary:
            trust_radius = min(3.0 * trust_radius, max_trust_radius)
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2.0 * trust_radius, max_trust_radius)
        # Accept proposed step if agreement exceeds threshold eta
        if rho > eta:
            p = p_prop
            fun = fun_prop
            jac = jac_prop
            jac_mag = np.sqrt(np.sum(jac * jac))
        k += 1
    return p, k, bool(jac_mag < gtol)


_LU = dict(diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))


class _LinearSolver:
    """The Newton systems of one solve (grounded active Laplacians), each by SuperLU or by PyAMG-preconditioned CG.

    LU unless the system has at least n0 rows and SuperLU's factors would hold at least fill times the nonzeros of
    the system (lower half). That fill is pre-screened in Reverse Cuthill-McKee order and, when borderline, counted
    exactly in SuperLU's own MMD order (from a near-zero-fill spilu), and recounted only once the system has changed
    by more than delta since the last count; in between the last choice stands. CG runs to rtol on a
    smoothed-aggregation hierarchy with a deterministic prolongation smoother; a system it does not solve to rtol
    falls back to LU, and so do the next ones until the next count."""

    def __init__(self, n0=3000, fill=40.0, delta=0.05, rtol=1e-8):
        self.n0, self.fill, self.delta, self.rtol = n0, fill, delta, rtol
        self.ref, self.amg = None, False
        self._lu_cache = (None, None)
        self._amg_cache = (None, None, None)
        self.counts = {"lu": 0, "amg": 0}

    def solve(self, A, b):
        n = A.shape[0]
        if n == 0:
            return np.zeros(0)
        order = None
        # Assess factorization strategy if system size exceeds threshold
        if n >= self.n0 and (self.ref is None or abs(A.nnz / self.ref[0] - 1) > self.delta
                             or abs(n / self.ref[1] - 1) > self.delta):
            # Screen Cholesky factor fill-in budget in Reverse Cuthill-McKee order
            lower = (A.nnz + n) / 2
            budget = int(np.ceil(self.fill * lower / 2 - n))
            indptr, indices = A.indptr.astype(np.int64), A.indices.astype(np.int64)
            rcm = csg.reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
            cnt_rcm = _count_lower(indptr, indices, rcm, budget)
            if cnt_rcm <= budget:
                self.amg = False
            else:
                # Count fill-in in SuperLU's MMD permutation
                order = np.argsort(spla.spilu(A.tocsc(), drop_tol=1e10, fill_factor=1, permc_spec="MMD_AT_PLUS_A",
                                              **_LU).perm_c)
                count = _count_lower(indptr, indices, order, budget)
                self.amg = count > budget
            self.ref = (A.nnz, n)
            # Factorize with SuperLU if sparsity budget allows
            if not self.amg and order is not None:
                self.counts["lu"] += 1
                lu = spla.splu(A[order][:, order].tocsc(), permc_spec="NATURAL", **_LU)
                def _solve_ord(rhs, _lu=lu, _ord=order):
                    out = np.empty(len(_ord))
                    out[_ord] = _lu.solve(rhs[_ord])
                    return out
                self._lu_cache = (A, _solve_ord)
                return _solve_ord(b) if b.any() else np.zeros(n)
        # Attempt PyAMG-preconditioned CG for dense factor systems
        if n >= self.n0 and self.amg:
            x = self._amg(A, b)
            if x is not None:
                self.counts["amg"] += 1
                return x
            self.amg = False                     # LU until the next count
        # Default: solve via SuperLU sparse factorization
        self.counts["lu"] += 1
        if not b.any():
            return np.zeros(n)
        if self._lu_cache[0] is not A:
            self._lu_cache = (A, spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", **_LU).solve)
        return self._lu_cache[1](b)

    def _amg(self, A, b):
        import pyamg

        if A.nnz >= 2**31:
            return None
        b2 = float((b * b).sum())
        if b2 == 0.0:
            return np.zeros_like(b)
        # Construct or reuse smoothed aggregation AMG preconditioner
        if self._amg_cache[0] is A:
            _, A32, M = self._amg_cache
        else:
            A32 = sp.csr_matrix((A.data, A.indices.astype(np.int32), A.indptr.astype(np.int32)), shape=A.shape)
            ml = pyamg.smoothed_aggregation_solver(A32, symmetry="hermitian", max_coarse=500,
                                                   smooth=("jacobi", {"omega": 4.0 / 3.0, "weighting": "local"}))
            M = ml.aspreconditioner()
            self._amg_cache = (A, A32, M)
        # Execute preconditioned Conjugate Gradient iterations
        x = np.zeros_like(b)
        r = b.copy()
        tol2 = (self.rtol ** 2) * b2
        z = M(r)
        p = z.copy()
        rz = float((r * z).sum())
        for _ in range(1000):
            Ap = A32 @ p
            pAp = float((p * Ap).sum())
            if pAp <= 0.0:
                break
            alpha = rz / pAp
            x += alpha * p
            r -= alpha * Ap
            if float((r * r).sum()) <= tol2:
                break
            z = M(r)
            rz_new = float((r * z).sum())
            p = z + (rz_new / rz) * p
            rz = rz_new
        # Check residual convergence tolerance
        res = b - A32 @ x
        return x if float((res * res).sum()) <= 100.0 * tol2 else None


def _auto_tau(tail, head, n):
    """tau = 0.3 times the cells in the largest connected piece of the graph (the benchmark's rule)."""
    if n == 0:
        return 1.0
    _, lab = csg.connected_components(sp.coo_matrix((np.ones(len(tail)), (tail, head)), shape=(n, n)),
                                      directed=False)
    return 0.3 * float(np.bincount(lab).max())


def _validate_tau(tau):
    """tau as a float, or "auto"; positivity and finiteness are checked by _solve_qrot."""
    if isinstance(tau, str):
        if tau != "auto":
            raise ValueError(f'tau must be a positive number or "auto", got {tau!r}.')
        return tau
    try:
        arr = np.asarray(tau, dtype=float)
    except (TypeError, ValueError):
        arr = None
    if arr is None or arr.size != 1:
        raise ValueError(f'tau must be a positive number or "auto", got {tau!r}.')
    return float(arr.item())


def _check_tau(tau, tail, head, n):
    """tau as a float: "auto" resolved by _auto_tau."""
    tau = _validate_tau(tau)
    if tau == "auto":
        tau = _auto_tau(tail, head, n)
        _log.detail(f"tau='auto': {tau:g} (0.3 x nodes in the largest connected piece)")
    return tau


def _solve_qrot(tail, head, c, f, tau, gtol=1e-4, tr_maxiter=100, tol=1e-12, eps=1e-8, max_iter=3000,
                linear_solver=None):
    """Optimal flow J and potentials p of the QROT dual (see qrot)."""
    # Validate regularization and cost parameters
    if not (np.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be a positive finite number, got {tau!r}.")
    c = np.asarray(c, dtype=float)
    if not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("Edge costs must be finite and nonnegative.")
    n, m = len(f), len(tail)
    scale = np.abs(f).max()

    # Demand must balance on every graph piece
    _, piece = csg.connected_components(sp.coo_matrix((np.ones(m), (tail, head)), shape=(n, n)), directed=False)
    if scale == 0 or np.any(np.abs(np.bincount(piece, f)) > tol * scale * np.bincount(piece)):
        raise ValueError("The demand (sinks minus sources) is zero or does not balance on a connected piece "
                         "of the graph.")

    # Primal edge flow function from dual potentials
    def flow(p):
        return np.maximum(p[head] - p[tail] - c, 0.0) / tau

    # Kirchhoff residual function: net imbalance per node
    def residual(J):
        return f - np.bincount(head, J, minlength=n) + np.bincount(tail, J, minlength=n)

    # Set up tau continuation ladder near the LP limit (tau max|f| < max c / 10)
    k = max(np.ceil(np.log10(0.1 * c.max() / (tau * scale))), 0) if c.max() > 0 else 0
    taus = tau * 10.0 ** np.arange(k, -1, -1)
    tau = float(taus[0])

    # Initialize potentials using shortest-path distance from sources
    G = sp.csr_matrix((c, (tail, head)), shape=(n, n))
    dist = csg.dijkstra(G, indices=np.flatnonzero(f < 0), min_only=True)
    reached = np.isfinite(dist)
    p = np.where(reached, dist, dist[reached].max())

    # Start trust-region Newton-CG warm start
    if tr_maxiter > 0:
        p, tr_nit, tr_ok = _trust_ncg(p, tail, head, c, f, tau, float(scale), eps, gtol, tr_maxiter)
    else:
        tr_nit, tr_ok = 0, False

    # Start Newton iterations across tau continuation levels
    linear_solver = _LinearSolver() if linear_solver is None else linear_solver
    steps = []
    for tau in taus:
        tau = float(tau)
        steps.append(0)
        prev_a = None
        for _ in range(max_iter):
            # Compute edge slacks and Kirchhoff residual; check convergence
            s, r, converged = _slack_residual(p, tail, head, c, f, tau, tol * scale)
            if converged:
                break

            # Assemble Hessian: Laplacian of active edges (s > -eps)
            a = s > -eps
            if prev_a is None or not np.array_equal(a, prev_a):
                prev_a = a
                A = sp.coo_matrix((np.full(a.sum(), 1.0 / tau), (tail[a], head[a])), shape=(n, n))
                L = csg.laplacian((A + A.T).tocsr()).tocsr()
                k, lab = csg.connected_components(L, directed=False)
                size = np.bincount(lab)
                free = _grounded_free(np.flatnonzero(L.diagonal() > 0), lab, k)
                L_free = L[1:, 1:] if k == 1 and free.size == n - 1 and free[0] == 1 else L[free][:, free]

            # Solve Newton system for free potentials grounded at one cell per piece
            mean = np.bincount(lab, r) / size
            mean[np.abs(mean) <= tol * scale] = 0.0
            d = np.zeros(n)
            d[free] = linear_solver.solve(L_free, (r - mean[lab])[free])

            # Shift disconnected pieces with net residual until a boundary edge opens
            if mean.any():
                cut = lab[tail] != lab[head]
                gap_in, gap_out = np.full(k, np.inf), np.full(k, np.inf)
                np.minimum.at(gap_in, lab[head[cut]], -s[cut])
                np.minimum.at(gap_out, lab[tail[cut]], -s[cut])
                move = np.where(mean > 0, gap_in, np.where(mean < 0, -gap_out, 0.0)) + tau * mean * size
                move = np.where(np.isfinite(move), move, 0.0) - np.where(mean != 0, np.bincount(lab, d) / size, 0.0)
                d += move[lab]

            # Execute Armijo line search on exact dual gain and update potentials
            p = p + _armijo_step(s, d[head] - d[tail], float((r * d).sum()), tau) * d
            steps[-1] += 1
        else:
            # Check LP feasibility if Newton loop failed to converge
            B = sp.coo_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[tail, head], np.r_[np.arange(m), np.arange(m)])), shape=(n, m))
            if linprog(np.zeros(m), A_eq=B, b_eq=f / scale, method="highs").status == 2:
                raise ValueError("The demand is infeasible: along the edge directions, the sources' mass cannot reach the sinks.")
            raise RuntimeError(f"QROT did not converge: Kirchhoff residual {np.abs(r).max():.2e} after {max_iter} Newton steps")

    # Zero unresolvable flows below tolerance
    J = flow(p)
    J[J <= tol * scale + np.finfo(float).eps * (np.abs(p[head]) + np.abs(p[tail])) / tau] = 0.0
    off = np.abs(residual(J)).max()
    _log.detail(f"solver: trust region {tr_nit} steps ({'converged' if tr_ok else 'capped'}), Newton {sum(steps)} "
                f"steps over {len(steps)} tau levels (max {max(steps)} per level; {linear_solver.counts['lu']} LU, "
                f"{linear_solver.counts['amg']} AMG-CG), Kirchhoff residual {off:.1e}")
    if off > 1e-6 * scale:
        warnings.warn(f"The flow misses the demand by {off / scale:.0e} of its size, below the solver's "
                      "resolution: raise tau, or scale the costs and masses toward 1.",
                      RuntimeWarning, skip_file_prefixes=(_PKG_DIR,))
    # Center potentials on unconstrained pieces and return flow and potentials
    return J, _midpoint_free_potentials(tail, head, c, J, p, tau, piece=piece)


def _midpoint_free_potentials(tail, head, c, J, p, tau, piece=None):
    """Potentials the flow leaves free, at the midpoint of their optimal range; NaN where unbounded."""
    n, on = len(p), J > 0
    if not on.any():
        return np.full(n, np.nan)

    # Build slack graph with zero-slack reverse edges along active flow
    t, h = np.r_[tail, head[on]], np.r_[head, tail[on]]
    s = np.r_[np.maximum(c + tau * J - (p[head] - p[tail]), 0.0), np.zeros(on.sum())]
    o = np.lexsort((s, h, t))
    o = o[np.r_[True, (np.diff(t[o]) != 0) | (np.diff(h[o]) != 0)]]
    G = sp.csr_matrix((s[o], (t[o], h[o])), shape=(n, n))

    # Pick one reference node per connected flow component piece
    cells = np.unique(np.r_[tail[on], head[on]])
    _, group = csg.connected_components(sp.coo_matrix((np.ones(on.sum()), (tail[on], head[on])), shape=(n, n)), directed=False)
    if piece is None:
        _, piece = csg.connected_components(sp.coo_matrix((np.ones(len(tail)), (tail, head)), shape=(n, n)), directed=False)
    ref = cells[np.isin(group[cells], group[cells[np.unique(piece[cells], return_index=True)[1]]])]

    # Compute earliest and latest admissible dual potential bounds via shortest paths
    up = csg.dijkstra(G, indices=ref, min_only=True)
    down = csg.dijkstra(G.T.tocsr(), indices=ref, min_only=True)
    # Center unconstrained potentials at the midpoint of their admissible interval
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


@_log.stage("solving quadratically regularized transport")
def qrot(
    graph_result: dict,
    adata=None,
    tau: float | str = "auto",
    normalize_costs: bool = True,
    cost_range: tuple = (1e-6, 1.0),
    verbose: int | bool = 1,
    reconnect: bool = True,
    reconnect_graph: str = "expr",
    graph: str | None = None,
    key_added: str | None = None,
) -> None:
    """Solve quadratically regularized optimal transport and compute pseudotime.

    Computes flow `J` along graph edges satisfying the source-sink demand `f`
    constructed by :func:`deltatrace.tl.endpoints`:

        min_{J >= 0}   <c, J> + (tau / 2) ||J||^2
        s.t.           B^T J = f = b - a

    where `B` is the cell-edge incidence matrix and `c` are edge costs. The dual
    potentials `p` satisfy the slackness relation ``J = (B p - c)_+ / tau``.
    Scaled to ``[0, 1]``, the potentials define continuous cellular pseudotime.

    Parameters
    ----------
    graph_result : dict
        Flow problem dictionary from :func:`deltatrace.tl.endpoints`.
    adata : anndata.AnnData, optional
        Target AnnData to write results into. Writes ``obs["pseudotime"]``,
        ``obs["pseudotime_raw"]``, and ``uns["dt_flow"]`` (containing edge flow array
        `J`, edge index pairs `edges`, and `tau`).
    tau : float or "auto", default "auto"
        Quadratic regularization parameter. ``"auto"`` sets tau to 0.3 times the
        number of nodes in the graph's largest connected piece.
    normalize_costs : bool, default True
        Whether to min-max scale edge costs into `cost_range` prior to solving.
    cost_range : tuple of float, default (1e-6, 1.0)
        Target range for normalized edge costs.
    verbose : int or bool, default 1
        Verbosity level (0: silent, 1: stages, 2: detailed progress).
    reconnect : bool, default True
        Give off-LCC cells a pseudotime via nearest-LCC reconnection.

        WARNING: with ``reconnect=False`` those cells are instead set to the
        median of the LCC cells' pseudotime -- an explicit placeholder, NOT an
        inferred position. Identify them via ``endpoints``' ``disconnected_info``
        before reading anything into it.
    reconnect_graph : str, default "expr"
        Graph key used for reconnecting off-LCC cells (typically ``"expr"``).
    graph : str, optional
        Graph name the endpoints were sampled on. Auto-detected if None.
    key_added : str, optional
        Key suffix for scoping multiple runs (e.g. ``obs["pseudotime_<key_added>"]``).

    Returns
    -------
    None
        Results are written directly into `adata`.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> gr = dt.tl.endpoints(adata, ["progenitor"], ["fate_A", "fate_B"])
    >>> dt.tl.qrot(gr, adata=adata, tau="auto")
    >>> "pseudotime" in adata.obs and "dt_flow" in adata.uns
    True
    """
    _sfx = _suffix(key_added)

    # Extract flow problem graph, costs, and demand from endpoints
    G = graph_result["G"]
    c = graph_result["cost_vec"]
    f = graph_result["f"]

    # Resolve target graph from AnnData if provided
    if adata is not None:
        if isinstance(adata, str):
            import scanpy as sc
            adata = sc.read_h5ad(adata)
        graph = _resolve_graph(adata, graph)

    # Extract edge index arrays and node mapping
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

    # Normalize edge costs into cost_range
    if normalize_costs and m > 0:
        c = _normalize_costs(c, cost_range)

    # Unpack edge tails u and heads v
    if node_map is None:
        us, vs = E[:, 0], E[:, 1]
    else:
        us = np.array([node_map[u] for u, _ in edges], dtype=np.int64)
        vs = np.array([node_map[v] for _, v in edges], dtype=np.int64)

    # Validate demand vector shape and align with internal indices
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

    # Resolve regularization parameter and solve QROT
    tau = _check_tau(tau, us, vs, n)
    J_opt, p = _solve_qrot(us, vs, c, f_vec, tau)

    _log.detail(f"QROT solved: {m} edges, tau={tau:g}")
    _log.detail(f"  Flow range: [{J_opt.min():.4e}, {J_opt.max():.4e}]")
    _log.detail(f"  Potential range: [{np.nanmin(p):.4e}, {np.nanmax(p):.4e}]")

    # Map raw potentials to cell names
    idx_to_name = G.graph["idx_to_name"]
    if node_map is None:
        pt_raw = {idx_to_name[i]: p[i] for i in range(n)}
    else:
        pt_raw = {idx_to_name[node]: p[i] for node, i in node_map.items()}

    # Min-max scale potentials onto [0, 1] interval to define pseudotime
    _vals = np.array(list(pt_raw.values()))
    _pmin = np.nanmin(_vals)
    _rng = (np.nanmax(_vals) - _pmin) or 1.0
    pt_norm = {k: (v - _pmin) / _rng for k, v in pt_raw.items()}

    # Store results in AnnData
    if adata is not None:
        obs_names = adata.obs_names.astype(str)
        adata.obs[f"pseudotime_raw{_sfx}"] = (
            pd.Series(pt_norm).reindex(obs_names).to_numpy(float)
        )

        # Reconnect off-LCC cells or assign median placeholder
        if reconnect:
            pt_full = _reconnect_pseudotime(
                adata,
                pt_norm,
                graph=graph,
                reconnect_graph=reconnect_graph,
            )
        else:
            pt_full = _median_off_lcc_pseudotime(adata, pt_norm, graph=graph)

        # Write finalized pseudotime column and flow dictionary
        adata.obs[f"pseudotime{_sfx}"] = (
            pd.Series(pt_full).reindex(obs_names).to_numpy(float)
        )
        if node_map is None:
            # np.array of an empty edge list is shape (0,), not (0, 2): keep that
            edges_arr = E.copy() if m else np.array([], dtype=np.int64)
        else:
            edges_arr = np.array(edges, dtype=np.int64)
        adata.uns[f"dt_flow{_sfx}"] = {
            "J": J_opt,
            "edges": edges_arr,
            "tau": float(tau),
        }


@_log.stage("solving quadratically regularized transport")
def qrot_graph(
    graph,
    a,
    b,
    tau: float | str = 100,
    directed: bool = False,
    normalize_costs: bool = True,
    verbose: int | bool = 1,
) -> dict:
    """Solve quadratically regularized optimal transport on a SciPy csgraph.

    The standalone solver behind :func:`qrot`, independent of AnnData. Minimizes:

        min_{J >= 0}   <c, J> + (tau / 2) ||J||^2
        s.t.           inflow - outflow = b - a

    Parameters
    ----------
    graph : (n, n) scipy.sparse array or matrix, or numpy.ndarray
        Edge costs in SciPy csgraph format: ``graph[i, j]`` is the cost of an edge
        from i to j. Self-loops are ignored.
    a, b : (n,) array_like
        Non-negative source (`a`) and sink (`b`) mass vectors. Must satisfy
        ``sum(a) == sum(b)``.
    tau : float or "auto", default 100
        Regularization strength. ``"auto"`` sets tau to 0.3 times the number of
        nodes in the largest connected component.
    directed : bool, default False
        Whether the input graph is directed. ``False`` treats edges as bidirectional.
    normalize_costs : bool, default True
        Scale edge costs by dividing by their maximum before solving.
    verbose : int or bool, default 1
        Verbosity level.

    Returns
    -------
    dict
        Dictionary containing:
        - ``"flow"``: (n, n) scipy.sparse.csr_array with optimal edge flows.
        - ``"potential"``: (n,) optimal dual potentials.
        - ``"pseudotime"``: (n,) potentials shifted to 0 and normalized to [0, 1].
        - ``"tau"``: the tau value used in the solve.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> import deltatrace as dt
    >>> # Line graph: 0 -> 1 -> 2
    >>> adj = sp.csr_matrix([[0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]])
    >>> a = np.array([1.0, 0.0, 0.0])
    >>> b = np.array([0.0, 0.0, 1.0])
    >>> res = dt.tl.qrot_graph(adj, a, b, tau=1.0, directed=True)
    >>> res["flow"].nnz > 0
    True
    """
    # Validate and standardize input graph format
    G = graph if sp.issparse(graph) or np.ma.isMaskedArray(graph) else np.asarray(graph, dtype=float)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"qrot_graph: graph must be square (n, n), got shape {G.shape}.")
    if not sp.issparse(G):      # SciPy's dense conventions: masked entries, or 0, inf and NaN, are no edge
        G = csg.csgraph_from_masked(G) if np.ma.isMaskedArray(G) else csg.csgraph_from_dense(G)
    G = sp.coo_array(sp.csr_array(G))
    n = G.shape[0]

    # Validate source and sink masses and ensure global balance
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (n,) or b.shape != (n,):
        raise ValueError(f"qrot_graph: a and b need one entry per node ({n}), got {a.shape} and {b.shape}.")
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or (a < 0).any() or (b < 0).any():
        raise ValueError("qrot_graph: a and b must be finite and nonnegative.")
    if a.sum() <= 0 or not np.isclose(a.sum(), b.sum(), rtol=1e-6, atol=0):
        raise ValueError(f"qrot_graph: a and b must have the same positive total, got {a.sum()} and {b.sum()}.")
    b = b * (a.sum() / b.sum())     # equal totals to rounding, as the solver's balance check needs

    # Extract edge list, ignore self-loops, and process edge directions
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

    # Resolve tau and solve optimal transport problem
    tau = _check_tau(tau, tail, head, len(a))
    J, p = _solve_qrot(tail, head, c, b - a, tau)
    _log.detail(f"QROT solved: {c.size} edges, tau={tau:g}")

    # Build sparse solution flow array
    flow = sp.csr_array((J, (tail, head)), shape=(n, n))
    flow.eliminate_zeros()

    # Shift component potentials to 0 and normalize to [0, 1] pseudotime
    _, piece = csg.connected_components(sp.coo_array((np.ones(c.size), (tail, head)), shape=(n, n)),
                                        directed=False)
    start = np.full(piece.max() + 1, np.inf)
    np.fmin.at(start, piece, p)
    pt = p - start[piece]
    span = np.nanmax(pt) if np.isfinite(pt).any() else 0.0
    return {"flow": flow, "potential": p, "pseudotime": pt / span if span > 0 else pt, "tau": float(tau)}


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
    """Project cellular transport flows onto a regular 2D velocity grid.

    Calculates net velocity vectors per cell by weighting embedding displacements
    by net transport flows:

        net_flow(u, v) = J(u -> v) - J(v -> u)

    Cell-level velocities are then smoothed onto a regular 2D grid using
    Gaussian-weighted kNN kernel interpolation.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData holding ``uns["dt_flow"]`` (from :func:`deltatrace.tl.qrot`) and
        embedding coordinates in ``obsm[basis]`` or ``obsm[f"X_{basis}"]``.
    basis : str, default "umap"
        Key identifying the 2D coordinate basis in `adata.obsm`.
    grid_size : tuple of int, default (25, 25)
        Number of grid points along each coordinate axis (nx, ny).
    k : int, default 25
        Number of nearest cells to interpolate per grid point.
    sigma : float, optional
        Bandwidth for Gaussian smoothing. Defaults to 0.8 * hypot of grid step.
    data_radius_factor : float, default 0.7
        Masks grid points farther than ``data_radius_factor * sigma`` from any cell.
    mag_transform : {"log1p", "asinh", "none"}, default "log1p"
        Transform applied to arrow magnitudes for visual contrast.
    transform_scale : float, optional
        Scale parameter for magnitude transform. Defaults to 80th percentile.
    clip_quantile : float, default 0.99
        Upper quantile clipping threshold for velocity magnitudes.
    edge_threshold : float, default 0
        Minimum net flow required for an edge to contribute to cell velocity.
    verbose : int or bool, default 1
        Verbosity level.
    flow_key : str, optional
        Specific key in `adata.uns` holding the flow dictionary.
    key_added : str, optional
        Suffix matching the one given to :func:`deltatrace.tl.qrot`.

    Returns
    -------
    None
        Writes ``adata.uns["dt_grid"]`` with interpolated grid components
        (``Xg``, ``Yg``, ``U``, ``V``, ``mask``, ``basis``) ready for
        :func:`deltatrace.pl.velocity` or :func:`deltatrace.pl.pseudotime`.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> gr = dt.tl.endpoints(adata, ["progenitor"], ["fate_A", "fate_B"])
    >>> dt.tl.qrot(gr, adata=adata)
    >>> dt.tl.compute_velocity_grid(adata, basis="spatial")
    >>> "dt_grid" in adata.uns
    True
    """
    from scipy.spatial import cKDTree

    _sfx = _suffix(key_added)
    flow_key = flow_key or f"dt_flow{_sfx}"
    # Keep the user's transform_scale so the raw per-cell arrows below honor it too.
    user_transform_scale = transform_scale

    # Retrieve flow data from uns
    flow = adata.uns.get(flow_key)
    if flow is None:
        raise ValueError(
            f"No flow data in adata.uns['{flow_key}']. Run qrot() with adata set "
            f"(and the same key_added) first."
        )
    edges = flow["edges"]
    J = flow["J"]

    # Extract 2D embedding coordinates
    coords = _resolve_basis(adata, basis)
    if not isinstance(coords, np.ndarray):       # e.g. a DataFrame in obsm
        coords = np.asarray(coords, dtype=float)

    obs_names = adata.obs_names.to_numpy()
    n = len(obs_names)
    J = np.asarray(J, dtype=float)
    nx_grid, ny_grid = grid_size

    # Compute net resultant flow velocity vector per cell
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

    # Set up regular 2D grid covering embedding bounding box
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_edges = np.linspace(x_min, x_max, nx_grid + 1)
    y_edges = np.linspace(y_min, y_max, ny_grid + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xg, Yg = np.meshgrid(x_centers, y_centers)

    if sigma is None:
        sigma = 0.8 * np.hypot((x_max - x_min) / nx_grid, (y_max - y_min) / ny_grid)

    # Query k nearest cells for all grid points using KD-tree
    tree = cKDTree(coords)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])  # (ny*nx, 2)
    k_eff = min(k, n)
    dists_flat, idxs_flat = tree.query(grid_pts, k=np.arange(1, k_eff + 1))

    # Smooth velocities onto grid points using Gaussian kernel weights
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

    # Compress velocity arrow magnitudes for visual clarity and clip outliers
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
    """What :func:`trajectory` returns."""

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
    it is stored, else the only graph there is (e.g. ``pp.integrate(adata)`` without a
    distance matrix stores just ``"expr"``)."""
    if graph is not None:
        return graph
    if "neighbors_intr_distances" in adata.obsp:
        return "intr"
    graph = _resolve_graph(adata, None)
    _log.detail(f"graph=None: no 'intr' graph stored, running on '{graph}'")
    return graph


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
    tau: float | str = "auto",
    seed: int = 42,
    edge_threshold: float = 0,
    compute_velocity: bool = True,
    omega_term: float | None = None,
    n_lineages: int | None = None,
    sink_weights: str = "uniform",
    verbose: int | bool = 1,
    key_added: str | None = None,
) -> dict:
    """Execute the complete DeltaTrace trajectory inference pipeline in one call.

    Sequentially executes:
    1. :func:`deltatrace.tl.endpoints`: samples source and sink endpoint cells.
    2. :func:`deltatrace.tl.qrot`: solves regularized transport and infers pseudotime.
    3. :func:`deltatrace.tl.compute_velocity_grid`: interpolates 2D velocity vectors (optional).
    4. :func:`deltatrace.tl.lineages`: computes absorbing Markov chain fate probabilities.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData containing integrated graph(s) from :func:`deltatrace.pp.integrate`
        and cluster annotations in ``obs[cluster_key]``.
    source_clusters : sequence of str
        Progenitor cluster label(s).
    sink_clusters : sequence of str, optional
        Terminal cluster label(s). If None, runs in source-only mode.
    graph : str, optional
        Graph name to run on (e.g. ``"intr"`` or ``"expr"``). Defaults to ``"intr"``
        when available.
    cluster_key : str, default "clusters"
        Column in ``adata.obs`` with cluster labels.
    p_end : float, default 0.15
        Endpoint sampling fraction.
    p_basis : {"cluster", "lcc"}, default "cluster"
        Sampling allocation basis across clusters.
    alloc_power : float, default 1.0
        Cluster quota scaling exponent.
    sampling_mode : str or dict, default "centered"
        Endpoint sampling strategy (``"centered"`` or ``"random"``).
    directed_metric : {"cost", "hops"}, default "cost"
        Metric used for geodesic distances in centered sampling.
    tau : float or "auto", default "auto"
        QROT regularization strength.
    reconnect_graph : str, default "expr"
        Graph key used to reconnect off-LCC disconnected cells.
    basis : str, optional
        Embedding coordinate key for velocity grid (e.g. ``"spatial"`` or ``"umap"``).
    seed : int, default 42
        Random seed for sampling.
    edge_threshold : float, default 0
        Minimum flow threshold for active edges in lineage Markov chain.
    compute_velocity : bool, default True
        Whether to compute the velocity grid field.
    omega_term : float, optional
        Terminality score cutoff for source-only lineage discovery.
    n_lineages : int, optional
        Number of lineages to discover in source-only mode.
    sink_weights : {"uniform", "cluster_balanced"}, default "uniform"
        Demand weight distribution in source-only mode.
    verbose : int or bool, default 1
        Verbosity level.
    key_added : str, optional
        Namespace suffix for written keys.

    Returns
    -------
    dict
        Dictionary containing ``"endpoints"`` and ``"lineages"`` results.
        Pseudotime, velocity grid, and lineage probabilities are written directly
        into `adata`. In source-only mode without a cutoff, returns ``"terminality"``
        ranking table.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> res = dt.tl.trajectory(
    ...     adata,
    ...     source_clusters=["progenitor"],
    ...     sink_clusters=["fate_A", "fate_B"],
    ...     basis="spatial",
    ... )
    >>> "pseudotime" in adata.obs and "lineage" in adata.obs
    True
    """
    # up front: a bad terminal-cut combination, tau or a missing embedding must not cost the
    # caller a full solve.
    _check_terminal_cut(sink_clusters, omega_term, n_lineages)
    tau = _validate_tau(tau)
    if tau != "auto" and not (np.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be a positive finite number, got {tau!r}.")
    source_clusters = _as_labels(source_clusters, "source_clusters")
    if sink_clusters is not None:
        sink_clusters = _as_labels(sink_clusters, "sink_clusters")
    _check_labels(adata, cluster_key, source_clusters=source_clusters,
                  sink_clusters=sink_clusters)
    graph = _trajectory_graph(adata, graph)
    if compute_velocity:
        basis = _velocity_basis(adata, basis)
    gr = endpoints(
        adata, source_clusters, sink_clusters,
        p_end=p_end, sampling_mode=sampling_mode,
        cluster_key=cluster_key, graph=graph, seed=seed, sink_weights=sink_weights,
        verbose=verbose, directed_metric=directed_metric, p_basis=p_basis,
        alloc_power=alloc_power,
    )
    qrot(
        gr, adata=adata, tau=tau, reconnect=True,
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
        tab = terminality(adata, show=bool(_log.resolve(verbose) > 0), **rank)
        if len(tab) == 0:
            raise ValueError(
                "Source-only run: no cluster receives flow from another cluster, so there "
                "are no fates to rank. Check source_clusters and cluster_key."
            )
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
