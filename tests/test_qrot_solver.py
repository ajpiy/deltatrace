"""The QROT dual solver (_solve_qrot) on the inputs that used to break it: invalid demand,
near-LP tau, the rounding floor, tiny sinks, cells and flow groups the optimum leaves free, and
one-sided bounds; then its public entry qrot_graph on SciPy graphs. Every problem is small and
generated inline."""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from deltatrace._transport import (
    _armijo_step,
    _count_lower,
    _LinearSolver,
    _normalize_costs,
    _solve_qrot,
)
from deltatrace.tl import qrot_graph


def _normalize(c):
    return 1e-6 + (1 - 1e-6) * (c - c.min()) / np.ptp(c)


def _knn(n, k, seed):
    """Symmetric kNN graph, both directions at one cost, min-maxed to [1e-6, 1] as qrot does."""
    X = np.random.default_rng(seed).random((n, 2))
    d, nb = cKDTree(X).query(X, k + 1)
    D = sp.coo_matrix((d[:, 1:].ravel(), (np.repeat(np.arange(n), k), nb[:, 1:].ravel())), shape=(n, n))
    D = D.tocsr().maximum(D.T.tocsr()).tocoo()
    return D.row.astype(np.int64), D.col.astype(np.int64), _normalize(D.data)


def _rgg(n, seed):
    """Random geometric graph below the connectivity radius (islands), endpoints in its LCC."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    pr = cKDTree(X).query_pairs(0.8 * np.sqrt(np.log(n) / (np.pi * n)), output_type="ndarray")
    w = _normalize(np.linalg.norm(X[pr[:, 0]] - X[pr[:, 1]], axis=1))
    t, h, c = np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]], np.r_[w, w]
    _, lab = csg.connected_components(sp.coo_matrix((c, (t, h)), shape=(n, n)), directed=False)
    k = n // 50
    pick = rng.choice(np.flatnonzero(lab == np.bincount(lab).argmax()), 2 * k, replace=False)
    return t, h, c, _demand(n, pick[:k], pick[k:])


def _demand(n, src, snk):
    f = np.zeros(n)
    f[src] = -1 / len(src)
    f[snk] = 1 / len(snk)
    return f


def _check_kkt(t, h, c, f, tau, J, p):
    """J >= 0, Kirchhoff residual within the solver's bound (1e-12 relative, or the rounding of
    the potentials), J > 0 exactly where the slack is tau J, slack <= 0 elsewhere."""
    n = f.size
    ok = ~np.isnan(p)
    s = p[h] - p[t] - c
    rnd = np.finfo(float).eps * (np.abs(np.nan_to_num(p[h])) + np.abs(np.nan_to_num(p[t])))
    w = np.where(s > -rnd, rnd / tau, 0.0)
    bound = 1e-12 * np.abs(f).max() + np.bincount(h, w, minlength=n) + np.bincount(t, w, minlength=n)
    r = f - np.bincount(h, J, minlength=n) + np.bincount(t, J, minlength=n)
    assert J.min() >= 0
    assert np.all(np.abs(r) <= 2 * bound)
    assert not np.isinf(p).any()
    de = ok[t] & ok[h]
    span = np.nanmax(p) - np.nanmin(p)
    on, off = (J > 0) & de, (J == 0) & de
    assert np.abs(s[on] - tau * J[on]).max() <= 1e-9 * span
    if off.any():
        assert s[off].max() <= 1e-9 * span


# --------------------------------------------------------------------- invalid demand
@pytest.mark.parametrize("tail, head, f", [
    ([0, 1], [1, 0], [0.0, 0.0]),                      # nothing to transport
    ([1, 2], [2, 1], [-1.0, 0.0, 1.0]),                # isolated source
    ([], [], [-1.0, 0.0, 1.0]),                        # no edges
    ([0, 1, 2, 3], [1, 0, 3, 2], [-1.0, 0.5, 0.0, 0.5]),  # balanced overall, not per piece
])
def test_unbalanced_demand_raises(tail, head, f):
    with pytest.raises(ValueError, match="does not balance on a connected piece"):
        _solve_qrot(np.array(tail, np.int64), np.array(head, np.int64), np.ones(len(tail)), np.array(f), 1.0)


@pytest.mark.parametrize("tail, head, f", [
    ([1], [0], [-1.0, 1.0]),                           # the only edge points the wrong way
    ([1, 1, 2], [0, 2, 1], [-1.0, 0.0, 1.0]),          # the source has no way out
    (np.arange(9), np.arange(1, 10), [-0.5, 0, 0, 0.9, 0, -0.5, 0, 0, 0, 0.1]),  # a one-way prefix short of mass
])
def test_infeasible_direction_raises(tail, head, f):
    with pytest.raises(ValueError, match="infeasible"):
        _solve_qrot(np.array(tail, np.int64), np.array(head, np.int64), np.full(len(tail), 0.5), np.array(f), 1.0)


# --------------------------------------------------------------------- newton
def test_tiny_sinks_get_their_demand():
    """Sinks of 1e-9, nine decades below the source, still receive exactly their demand."""
    t, h, c = [0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 0, 0, 0], [1e-6, 1.0, 0.5, 1e-6] * 2
    t, h, c = np.array(t), np.array(h), np.array(c)
    f = np.array([0, -1, 1 - 2e-9, 1e-9, 1e-9])
    J, p = _solve_qrot(t, h, c, f, 100.0)
    _check_kkt(t, h, c, f, 100.0, J, p)
    assert np.allclose(J[[1, 2, 3]], [1 - 2e-9, 1e-9, 1e-9], rtol=1e-6, atol=0)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("ratio", [1e-3, 1e-4, 1e-8])
def test_near_lp_tau(seed, ratio):
    """tau max|f| far below max c: few edges carry flow and Newton must find them."""
    t, h, c = _knn(200, 10, seed)
    pick = np.random.default_rng(seed).choice(200, 6, replace=False)
    f = _demand(200, pick[:3], pick[3:])
    tau = ratio * c.max() / np.abs(f).max()
    J, p = _solve_qrot(t, h, c, f, tau)
    _check_kkt(t, h, c, f, tau, J, p)


def test_islands_need_many_newton_steps():
    """Islands below the connectivity radius: Newton needs over 300 damped steps on one tau level
    here, so a small step cap would fail."""
    t, h, c, f = _rgg(3000, 1)
    J, p = _solve_qrot(t, h, c, f, 1.0)
    _check_kkt(t, h, c, f, 1.0, J, p)


def test_armijo_step_matches_the_dual_gain():
    """The compiled line search accepts the first t = 2^-k whose exact dual gain clears the Armijo bound."""
    rng = np.random.default_rng(0)
    m, tau = 500, 0.3
    s, dd = rng.normal(0, 1, m), rng.normal(0, 3, m)
    slope = 0.5 * np.abs(dd).sum()

    def gain(t):
        ds = t * dd
        s1 = s + ds
        term = np.where((s > 0) & (s1 > 0), -ds**2, 0.0)
        term = np.where((s > 0) & ~(s1 > 0), s * (2 * ds + s), term)
        term = np.where(~(s > 0) & (s1 > 0), -s1**2, term)
        return t * slope + term.sum() / (2 * tau)

    t = 1.0
    while gain(t) < 1e-4 * t * slope and t > 1e-12:
        t /= 2
    assert t < 1.0
    assert _armijo_step(s, dd, slope, tau) == t


def test_rounding_floor():
    """All costs 1, tau 1e-3 on a 30x30 grid: J = slack / tau is only known to ulp(p) / tau,
    above 1e-12 of the demand, so the solve must stop at the rounding floor."""
    g = 30
    idx = np.arange(g * g).reshape(g, g)
    i, j = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()], np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
    t, h, c = np.r_[i, j], np.r_[j, i], np.ones(2 * i.size)
    f = _demand(g * g, [0], [g * g - 1])
    J, p = _solve_qrot(t, h, c, f, 1e-3)
    _check_kkt(t, h, c, f, 1e-3, J, p)


# --------------------------------------------------------------------- centering
def test_package_regime_flow_matches_potentials():
    """tau 100 on a kNN graph: no flow is left on an edge that the midpoint closes."""
    t, h, c = _knn(300, 15, 3)
    pick = np.random.default_rng(3).choice(300, 10, replace=False)
    f = _demand(300, pick[:5], pick[5:])
    J, p = _solve_qrot(t, h, c, f, 100.0)
    _check_kkt(t, h, c, f, 100.0, J, p)


def test_floor_costs_near_lp():
    """All costs at the 1e-6 floor and tau 1e-3: every slack is ~1e-6."""
    t, h, c = _knn(600, 15, 0)
    pick = np.random.default_rng(0).choice(600, 10, replace=False)
    f = _demand(600, pick[:5], pick[5:])
    J, p = _solve_qrot(t, h, np.full_like(c, 1e-6), f, 1e-3)
    _check_kkt(t, h, np.full_like(c, 1e-6), f, 1e-3, J, p)


def _optimal_range(t, h, c, tau, J, p, v, ref):
    """Smallest and largest p_v over the optimal potentials, with the cells ref held at p (LP)."""
    n, on = p.size, J > 0
    rows = sp.csr_matrix((np.r_[np.ones(t.size), -np.ones(t.size)], (np.r_[np.arange(t.size), np.arange(t.size)], np.r_[h, t])),
                         shape=(t.size, n))
    bounds = [(p[i], p[i]) if i in ref else (None, None) for i in range(n)]
    out = []
    for sign in (1.0, -1.0):
        obj = np.zeros(n)
        obj[v] = sign
        r = linprog(obj, A_ub=rows[~on], b_ub=c[~on], A_eq=rows[on], b_eq=(c + tau * J)[on], bounds=bounds, method="highs")
        out.append(sign * r.fun)
    return out


@pytest.mark.parametrize("seed", [2, 15])
@pytest.mark.parametrize("tau", [1e-2, 1.0, 100.0])
def test_free_potentials_at_midpoint_of_their_range(seed, tau):
    """Every cell the flow leaves free, and every flow group but the reference one, sits at the
    midpoint of its earliest and latest optimal potential, found here by LP."""
    t, h, c, f = _rgg(120, seed)
    J, p = _solve_qrot(t, h, c, f, tau)
    _check_kkt(t, h, c, f, tau, J, p)
    on = J > 0
    _, group = csg.connected_components(sp.coo_matrix((np.ones(on.sum()), (t[on], h[on])), shape=(p.size, p.size)), directed=False)
    flow = np.unique(np.r_[t[on], h[on]])
    ref = set(np.flatnonzero(group == group[flow[0]]))
    span = np.nanmax(p) - np.nanmin(p)
    for v in np.flatnonzero(~np.isnan(p)):
        if v in ref:
            continue
        lo, hi = _optimal_range(t, h, c, tau, J, p, v, ref)
        assert p[v] == pytest.approx((lo + hi) / 2, abs=1e-8 * span)


def test_flow_groups_midpoint():
    """Two flow groups {0, 1} and {2, 3} in one piece: the flow fixes neither offset
    d = p2 - p1, so it goes to the midpoint of the range the free edges allow."""
    t, h = np.array([0, 2, 1, 2, 3, 0]), np.array([1, 3, 2, 1, 0, 3])
    c = np.array([0.5, 0.5, 1.0, 1.0, 3.0, 3.0])
    f = np.array([-1.0, 1.0, -1.0, 1.0])
    J, p = _solve_qrot(t, h, c, f, 1.0)
    _check_kkt(t, h, c, f, 1.0, J, p)
    # with p3 - p0 = d + 3: -1 <= d <= 1 (edges 1-2) and -6 <= d <= 0 (edges 3-0), so -1 <= d <= 0
    assert p[2] - p[1] == pytest.approx(-0.5, abs=1e-12)


# --------------------------------------------------------------------- NaN rule
def test_piece_without_demand_is_nan():
    t, h = np.array([0, 1, 2, 3]), np.array([1, 0, 3, 2])
    f = np.array([-1.0, 1.0, 0.0, 0.0])
    _, p = _solve_qrot(t, h, np.full(4, 0.5), f, 1.0)
    assert np.isfinite(p[:2]).all() and np.isnan(p[2:]).all()


@pytest.mark.parametrize("f, nan", [([-1.0, 1.0, 0.0], 2), ([0.0, -1.0, 1.0], 0)])
def test_one_sided_cell_is_nan(f, nan):
    """One-way chain 0 -> 1 -> 2 with the flow on one edge: the third cell is bounded on one
    side only, so it has no midpoint."""
    _, p = _solve_qrot(np.array([0, 1]), np.array([1, 2]), np.array([0.3, 0.7]), np.array(f), 1.0)
    assert np.isnan(p[nan]) and np.isfinite(np.delete(p, nan)).all()


# --------------------------------------------------------------------- qrot_graph
def test_qrot_graph_matches_the_solver():
    """A symmetric SciPy graph gives the solver's flow and potentials, with every entry an edge."""
    t, h, c = _knn(150, 8, 0)
    f = _demand(150, [0, 1, 2], [147, 148, 149])
    res = qrot_graph(sp.csr_array((c, (t, h)), shape=(150, 150)), np.maximum(-f, 0), np.maximum(f, 0),
                     normalize_costs=False, verbose=0)
    J, p = _solve_qrot(t, h, c, f, 100.0)
    Jg = res["flow"].toarray()[t, h]
    _check_kkt(t, h, c, f, 100.0, Jg, res["potential"])
    assert np.allclose(Jg, J, rtol=0, atol=1e-12 * J.max())
    assert np.allclose(res["potential"], p, rtol=0, atol=1e-9, equal_nan=True)


def test_qrot_graph_undirected_by_default():
    """A path stored one way (i -> i+1) still carries mass back from 3 to 0; directed=True cannot."""
    G = sp.csr_array((np.ones(3), ([0, 1, 2], [1, 2, 3])), shape=(4, 4))
    a, b = np.array([0, 0, 0, 1.0]), np.array([1.0, 0, 0, 0])
    res = qrot_graph(G, a, b, tau=1.0, verbose=0)
    assert np.allclose([res["flow"][3, 2], res["flow"][2, 1], res["flow"][1, 0]], 1)
    assert res["pseudotime"][3] == 0 and res["pseudotime"][0] == 1
    with pytest.raises(ValueError, match="infeasible"):
        qrot_graph(G, a, b, tau=1.0, directed=True, verbose=0)


def test_qrot_graph_dense_zero_is_no_edge():
    D = np.array([[0, 1.0, 0], [1.0, 0, 2.0], [0, 2.0, 0]])
    res = qrot_graph(D, [1.0, 0, 0], [0, 0, 1.0], verbose=0)
    assert res["flow"][0, 2] == 0 and np.isclose(res["flow"][0, 1], 1) and np.isclose(res["flow"][1, 2], 1)


def test_qrot_graph_explicit_sparse_zero_is_an_edge():
    G = sp.csr_array((np.array([0.0, 1.0]), ([0, 1], [1, 2])), shape=(3, 3))
    res = qrot_graph(G, [1.0, 0, 0], [0, 0, 1.0], tau=1.0, normalize_costs=False, verbose=0)
    assert np.isclose(res["flow"][0, 1], 1) and np.isclose(res["flow"][1, 2], 1)


def test_qrot_graph_pair_stored_both_ways_keeps_the_smaller_cost():
    """0 -> 1 costs 1 and 1 -> 0 costs 3: the rise along the flow is c + tau J = 1 + 1."""
    G = sp.csr_array((np.array([1.0, 3.0, 1.0]), ([0, 1, 1], [1, 0, 2])), shape=(3, 3))
    res = qrot_graph(G, [1.0, 0, 0], [0, 0, 1.0], tau=1.0, normalize_costs=False, verbose=0)
    assert np.isclose(res["potential"][1] - res["potential"][0], 2.0)


def test_qrot_graph_ignores_self_loops():
    t, h, c = _knn(60, 6, 1)
    f = _demand(60, [0], [59])
    G = sp.csr_array((c, (t, h)), shape=(60, 60))
    loops = sp.csr_array((np.full(60, 0.5), (np.arange(60), np.arange(60))), shape=(60, 60))
    one, two = (qrot_graph(g, np.maximum(-f, 0), np.maximum(f, 0), verbose=0) for g in (G, G + loops))
    assert np.array_equal(one["potential"], two["potential"], equal_nan=True)


def test_qrot_graph_piece_without_flow_is_nan():
    G = sp.csr_array((np.ones(2), ([0, 2], [1, 3])), shape=(4, 4))
    res = qrot_graph(G, [1.0, 0, 0, 0], [0, 1.0, 0, 0], verbose=0)
    assert np.isnan(res["pseudotime"][2:]).all() and np.isfinite(res["pseudotime"][:2]).all()
    assert res["flow"].nnz == 1


@pytest.mark.parametrize("graph, a, b, match", [
    (np.ones((2, 3)), [1.0, 0], [0, 1.0], "square"),
    (np.ones((2, 2)), [1.0, 0, 0], [0, 1.0], "one entry per node"),
    (np.array([[0, -1.0], [-1.0, 0]]), [1.0, 0], [0, 1.0], "nonnegative"),
    (sp.csr_array(np.array([[0, np.nan], [np.nan, 0]])), [1.0, 0], [0, 1.0], "finite"),
    (np.ones((2, 2)), [1.0, 0], [0, 2.0], "same positive total"),
    (np.ones((2, 2)), [1.0, -1.0], [0, 0.0], "nonnegative"),
    (np.eye(2), [1.0, 0], [0, 1.0], "no edges"),
])
def test_qrot_graph_rejects_bad_input(graph, a, b, match):
    with pytest.raises(ValueError, match=match):
        qrot_graph(graph, a, b, verbose=0)


@pytest.mark.parametrize("tau", [0, -1.0, np.nan, np.inf])
def test_qrot_graph_rejects_bad_tau(tau):
    with pytest.raises(ValueError, match="tau must be a positive finite number"):
        qrot_graph(np.ones((2, 2)), [1.0, 0], [0, 1.0], tau=tau, verbose=0)


@pytest.mark.parametrize("cost_range", [(-1.0, 1.0), (1.0, 0.0), (np.nan, 1.0), (0.0, np.inf)])
def test_cost_range_is_checked(cost_range):
    with pytest.raises(ValueError, match="cost_range"):
        _normalize_costs(np.array([1.0, 2.0, 3.0]), cost_range)


def test_qrot_graph_reads_dense_input_as_scipy_does():
    """A masked array keeps its unmasked zero as a zero-cost edge and drops its masked entries; in a
    plain dense array, inf and NaN are no edge."""
    D = np.array([[0, 0.0, 9, 0], [0.0, 0, 0, 1.0], [9, 0, 0, 1.0], [0, 1.0, 1.0, 0]])
    M = np.ma.masked_array(D, mask=[[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 1]])
    flow = qrot_graph(M, [1.0, 0, 0, 0], [0, 0, 1.0, 0], tau=1.0, normalize_costs=False, verbose=0)["flow"]
    assert flow[0, 2] == 0 and np.isclose(flow[0, 1], 1) and np.isclose(flow[3, 2], 1)
    D = np.array([[0, 1.0, np.inf], [1.0, 0, 1.0], [np.nan, 1.0, 0]])
    flow = qrot_graph(D, [1.0, 0, 0], [0, 0, 1.0], verbose=0)["flow"]
    assert flow[0, 2] == 0 and np.isclose(flow[1, 2], 1)


def test_qrot_graph_normalization_keeps_the_cheapest_path():
    """Direct 0 -> 2 costs 1.5, the detour through 1 costs 2: near the LP limit the direct edge carries
    the mass, with or without normalization (a min-max shift would favor the detour)."""
    G = sp.csr_array(([1.0, 1.0, 1.5], ([0, 1, 0], [1, 2, 2])), shape=(3, 3))
    for norm in (True, False):
        flow = qrot_graph(G, [1.0, 0, 0], [0, 0, 1.0], tau=1e-3, normalize_costs=norm, verbose=0)["flow"]
        assert np.isclose(flow[0, 2], 1) and flow[0, 1] == 0


def test_qrot_graph_accepts_float32_masses():
    a, b = np.array([0.1, 0.2, 0.7, 0], np.float32), np.array([0, 0, 0, 1], np.float32)
    G = sp.csr_array((np.ones(3), ([0, 1, 2], [1, 2, 3])), shape=(4, 4))
    assert np.isclose(qrot_graph(G, a, b, verbose=0)["flow"][2, 3], 1)


def test_qrot_graph_pseudotime_starts_every_piece_at_zero():
    """Two separate paths, 0-1-2 and 3-4-5-6: each source sits at 0, the longer path's sink at 1."""
    G = sp.csr_array((np.ones(5), ([0, 1, 3, 4, 5], [1, 2, 4, 5, 6])), shape=(7, 7))
    pt = qrot_graph(G, [0.5, 0, 0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0, 0, 0.5], verbose=0)["pseudotime"]
    assert pt[0] == 0 and pt[3] == 0 and pt[6] == 1 and 0 < pt[2] < 1


def test_qrot_graph_warns_below_the_solver_resolution():
    """Masses of 1e-15 at tau 1e-3: the flow cannot be resolved, and the solve says so."""
    P = sp.csr_array((np.arange(1.0, 10.0), (np.arange(9), np.arange(1, 10))), shape=(10, 10))
    with pytest.warns(RuntimeWarning, match="misses the demand"):
        qrot_graph(P, np.eye(10)[0] * 1e-15, np.eye(10)[9] * 1e-15, tau=1e-3, verbose=0)



# --------------------------------------------------------------------- linear solver: LU or AMG-CG per Newton step
def _knn_d(n, k, d, seed):
    """Symmetric kNN graph in d dimensions (dense active sets at large tau), costs min-maxed as qrot does."""
    X = np.random.default_rng(seed).random((n, d))
    dist, nb = cKDTree(X).query(X, k + 1)
    D = sp.coo_matrix((dist[:, 1:].ravel(), (np.repeat(np.arange(n), k), nb[:, 1:].ravel())), shape=(n, n))
    D = D.tocsr().maximum(D.T.tocsr()).tocoo()
    return D.row.astype(np.int64), D.col.astype(np.int64), _normalize(D.data)


def _grounded_laplacian(n, d, seed):
    t, h, _ = _knn_d(n, 10, d, seed)
    L = csg.laplacian(sp.coo_matrix((np.ones(t.size), (t, h)), shape=(n, n)).tocsr()).tocsr()
    return L[1:, 1:].tocsr()


@pytest.mark.parametrize("d", [2, 8])
def test_fill_count_is_superlus(d):
    """The fill counted in SuperLU's MMD order (from spilu) is the fill splu produces."""
    A = _grounded_laplacian(600, d, d)
    kw = dict(permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    order = np.argsort(sp.linalg.spilu(A.tocsc(), drop_tol=1e10, fill_factor=1, **kw).perm_c)
    count = _count_lower(A.indptr.astype(np.int64), A.indices.astype(np.int64), order, 10**12)
    lu = sp.linalg.splu(A.tocsc(), **kw)
    assert 2 * (count + A.shape[0]) == lu.L.nnz + lu.U.nnz
    assert _count_lower(A.indptr.astype(np.int64), A.indices.astype(np.int64), order, 10) == 11


def test_fill_count_matches_symbolic_elimination():
    rng = np.random.default_rng(3)
    for seed in range(10):
        W = sp.random(40, 40, 0.08, random_state=seed)
        A = (csg.laplacian(((W + W.T) > 0).astype(float)) + sp.eye(40)).tocsr()
        order = rng.permutation(40)
        B = A[order][:, order].toarray() != 0
        for k in range(40):
            nb = np.flatnonzero(B[k, k + 1:]) + k + 1
            B[np.ix_(nb, nb)] = True
        assert _count_lower(A.indptr.astype(np.int64), A.indices.astype(np.int64), order, 10**9) == np.tril(B, -1).sum()


@pytest.mark.parametrize("tau", [1.0, 300.0])
def test_amg_steps_give_the_lu_flow(tau):
    """Every Newton step by AMG-CG reaches the flow every step by LU reaches (the optimum is unique)."""
    t, h, c = _knn_d(800, 10, 8, 0)
    f = _demand(800, np.arange(10), np.arange(700, 800))
    lu, amg = _LinearSolver(n0=10**9), _LinearSolver(n0=0, fill=0.0)
    J1, p1 = _solve_qrot(t, h, c, f, tau, linear_solver=lu)
    J2, p2 = _solve_qrot(t, h, c, f, tau, linear_solver=amg)
    assert lu.counts["amg"] == 0 and amg.counts["amg"] > 0 and amg.counts["lu"] == 0
    for J, p in ((J1, p1), (J2, p2)):
        _check_kkt(t, h, c, f, tau, J, p)
    assert np.abs(J2 - J1).max() <= 1e-9 * J1.max()


def test_switch_picks_amg_only_for_dense_factors():
    """Large tau on a high-dimensional kNN graph makes the factors dense (AMG); small tau keeps them sparse (LU).
    At this size the fill stays below the default threshold, so the test lowers it."""
    t, h, c = _knn_d(3000, 15, 10, 1)
    f = _demand(3000, np.arange(20), np.arange(2900, 3000))
    dense, sparse = _LinearSolver(n0=1000, fill=10.0), _LinearSolver(n0=1000, fill=10.0)
    _solve_qrot(t, h, c, f, 900.0, linear_solver=dense)
    _solve_qrot(t, h, c, f, 1e-2, linear_solver=sparse)
    assert dense.counts["amg"] > 0 and sparse.counts["amg"] == 0


def test_amg_solve_is_deterministic_and_leaves_numpy_rng_alone():
    A = _grounded_laplacian(2000, 8, 5)
    b = np.random.default_rng(0).normal(size=A.shape[0])
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    x1, x2 = (_LinearSolver()._amg(A, b) for _ in range(2))
    assert np.array_equal(x1, x2)
    assert np.array_equal(np.random.get_state()[1], before)
    assert np.linalg.norm(A @ x1 - b) <= 1e-7 * np.linalg.norm(b)


# --------------------------------------------------------------------- tau="auto"
def test_auto_tau_is_0_3_times_the_largest_piece():
    """Pieces of 6 and 3 nodes: tau = 0.3 * 6, returned with the result."""
    G = sp.csr_array((np.ones(7), ([0, 1, 2, 3, 4, 6, 7], [1, 2, 3, 4, 5, 7, 8])), shape=(9, 9))
    a, b = np.r_[1.0, np.zeros(8)], np.r_[np.zeros(5), 1.0, np.zeros(3)]
    assert qrot_graph(G, a, b, tau="auto", verbose=0)["tau"] == pytest.approx(1.8)
    assert qrot_graph(G, a, b, verbose=0)["tau"] == 100.0


@pytest.mark.parametrize("tau", ["big", [1.0, 2.0], None])
def test_tau_that_is_not_a_number_or_auto_is_rejected(tau):
    with pytest.raises(ValueError, match="tau must be a positive"):
        qrot_graph(np.ones((2, 2)), [1.0, 0], [0, 1.0], tau=tau, verbose=0)


def test_size_one_array_tau_is_a_number():
    res = qrot_graph(np.ones((2, 2)), [1.0, 0], [0, 1.0], tau=np.array([2.0]), verbose=0)
    assert res["tau"] == 2.0


def test_failed_amg_solve_falls_back_to_lu_until_the_next_count():
    """A system CG cannot solve to rtol is factored, and the next similar one goes straight to LU."""
    A = _grounded_laplacian(2000, 8, 5)
    b = np.random.default_rng(0).normal(size=A.shape[0])
    ls = _LinearSolver(n0=0, fill=0.0, rtol=1e-30)          # unreachable rtol: every CG solve fails
    x = ls.solve(A, b)
    assert np.allclose(A @ x, b) and ls.counts == {"lu": 1, "amg": 0} and ls.amg is False
    calls = []
    ls._amg = lambda A, b: calls.append(1)
    ls.solve(A, b)
    assert calls == [] and ls.counts["lu"] == 2


@pytest.mark.parametrize("c", [[-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0]])
def test_solve_qrot_rejects_bad_costs(c):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _solve_qrot(np.array([0, 1]), np.array([1, 0]), np.array(c), np.array([-1.0, 1.0]), 1.0)


def test_trust_ncg_cg_cap_exit():
    """Inner CG loop exiting on max_cg cap must define step properly and produce finite potentials."""
    from deltatrace._transport import _trust_ncg

    t, h, c = _knn(50, 6, 42)
    f = _demand(50, [0], [49])
    p0 = np.zeros(50)
    p_out, k, ok = _trust_ncg(p0, t, h, c, f, 1.0, 1.0, maxiter=3, max_cg=1)
    assert np.isfinite(p_out).all()
    assert k > 0
    assert isinstance(ok, bool)
    assert not np.array_equal(p_out, p0)

