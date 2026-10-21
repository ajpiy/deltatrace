"""Power allocation: how ``alloc_power`` shares a role's total out over its clusters.

``p_basis="cluster"`` sizes each cluster from its own eligibility pool. ``alloc_power`` is
the exponent ``k`` in that sizing, quota proportional to ``pool_c ** k``, so the option spans
the two classical stratified-sampling rules and everything between them: ``k=1`` proportional
(the default and the historical path), ``k=0`` equal, ``k=0.5`` the square-root compromise.

Four properties are load-bearing and each has a test here.

``k=1`` is the old allocation. Not "agrees to the eye" but the pre-change quotas, asserted as
literals, with 2+ clusters on BOTH roles so both quota calls are exercised. The sampled
indices are pinned too, against the current tree rather than the pre-change one: per-(role,
cluster) RNG streams moved which cells a quota buys, and never how many.

The budget is the ``k=1`` total. ``p_end`` stays the size knob and ``k`` moves cells
between a role's clusters, so a sweep over ``k`` holds the sample size fixed up to the drift
the floor of 1 and the clamp to a pool can cause, in either direction.

Integerisation is largest-remainder with ties on the cluster NAME. That makes the quotas
independent of ``seed`` (unlike ``p_basis="lcc"``, whose remainder goes to a random few) and
independent of the ORDER the caller listed the clusters in. Both are tested because both are
the reason the parameter exists: a sweep over ``k`` is unreadable if the quotas also wander.

Floor of 1 and clamp to the pool come last and may move the realized total off the budget.
That drift is intended, and the sum test says so by holding only where neither binds.
"""

import ast
import contextlib
import io
import itertools

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from _line_toy import line_toy

import deltatrace as dt

SEEDS = (0, 1, 7, 42, 123, 2024)
KS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)

# Sampled indices with p_basis="cluster", alloc_power=1.0, pool object below,
# ["A", "B"] -> ["C", "D", "rare"], p_end=0.10. Two source clusters and three sink
# clusters, so every quota call is exercised.
#
# Re-captured when sampling moved to a per-(role, cluster) RNG stream, which retired the
# shared stream the original pre-change capture came off. The QUOTAS this config produces did
# not move with that change and are asserted as literals below; these tables pin the cells.
K1_RANDOM = {
    0: ([58, 22, 57, 45, 95, 19, 3, 33, 0, 52, 101, 115, 154, 151, 127, 110],
        [162, 193, 173, 182, 202, 204, 217]),
    1: ([69, 75, 37, 57, 66, 27, 28, 25, 4, 61, 141, 103, 106, 144, 139, 122],
        [174, 161, 160, 168, 202, 203, 218]),
    7: ([41, 9, 96, 42, 88, 82, 35, 92, 53, 39, 124, 134, 140, 146, 104, 155],
        [172, 179, 190, 167, 200, 201, 218]),
    42: ([21, 97, 38, 66, 68, 90, 63, 78, 30, 1, 123, 112, 152, 148, 129, 150],
         [194, 182, 183, 181, 203, 211, 218]),
    123: ([76, 42, 15, 45, 70, 60, 46, 52, 35, 75, 117, 131, 128, 106, 118, 104],
          [194, 160, 181, 189, 207, 211, 219]),
    2024: ([43, 0, 87, 21, 97, 85, 78, 6, 71, 53, 158, 115, 101, 147, 151, 103],
           [193, 186, 165, 162, 215, 214, 218]),
}
# Same tree, same shape, sampling_mode="directed" (medoid, anchor and shells all involved).
K1_DIRECTED = {
    0: ([6, 4, 5, 0, 1, 9, 7, 3, 2, 8, 103, 102, 105, 106, 104, 101],
        [197, 198, 196, 199, 215, 216, 219]),
    1: ([6, 7, 2, 0, 5, 1, 8, 3, 4, 9, 103, 104, 101, 102, 105, 100],
        [197, 198, 196, 199, 215, 216, 219]),
    7: ([6, 4, 0, 8, 3, 9, 2, 7, 1, 5, 103, 102, 101, 106, 100, 105],
        [197, 198, 196, 195, 215, 216, 219]),
    42: ([6, 0, 1, 9, 5, 3, 7, 8, 4, 2, 103, 106, 105, 100, 104, 101],
         [197, 196, 198, 195, 215, 216, 219]),
    123: ([6, 0, 9, 8, 3, 4, 2, 7, 1, 5, 103, 106, 101, 100, 105, 104],
          [197, 198, 196, 195, 215, 216, 219]),
    2024: ([6, 3, 7, 2, 8, 4, 1, 9, 0, 5, 103, 102, 104, 106, 105, 100],
           [197, 196, 198, 199, 215, 214, 219]),
}
# What k=1 asks of that config, unchanged since before alloc_power existed and seed-free
# under this basis: 10% of 100/60 on the sources, of 40/17/3 on the sinks.
K1_QUOTAS = ({"A": 10, "B": 6}, {"C": 4, "D": 2, "rare": 1})


def _pool_object(sizes, k=6):
    """Object whose clusters have exactly the given eligible-pool sizes.

    Cells sit on one line and are joined by a ``k``-nearest-neighbor graph, so the whole
    object is one connected component and every cluster's pool equals its size. That makes
    the expected quotas arithmetic rather than something to be measured.

    Parameters
    ----------
    sizes : dict
        Cluster label -> number of cells, in the order the cells are laid out.
    k : int, default 6
        Neighbors per cell in the hand-built graph.

    Returns
    -------
    anndata.AnnData
        Object carrying ``obs['clusters']`` and a ``neighbors_intr`` graph.
    """
    labels = [c for c, n in sizes.items() for _ in range(n)]
    n = len(labels)
    xy = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    D = np.abs(xy[:, None, 0] - xy[None, :, 0])
    np.fill_diagonal(D, np.inf)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in np.argsort(D[i])[:k]:
            rows.append(i)
            cols.append(int(j))
            vals.append(float(D[i, int(j)]))
    dist = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    conn = sp.csr_matrix((1.0 / (1.0 + np.asarray(vals)), (rows, cols)), shape=(n, n))
    obj = ad.AnnData(
        X=np.zeros((n, 2), dtype=np.float32),
        obs=pd.DataFrame({"clusters": labels}, index=[f"c{i}" for i in range(n)]),
    )
    obj.obsm["spatial"] = xy
    obj.obsp["neighbors_intr_distances"] = dist
    obj.obsp["neighbors_intr_connectivities"] = conn
    obj.uns["neighbors_intr"] = {
        "connectivities_key": "neighbors_intr_connectivities",
        "distances_key": "neighbors_intr_distances",
        "params": {"n_neighbors": k, "method": "umap"},
    }
    return obj


# One tiny cluster ("rare") and one small one ("D"), so the floor of 1 and the clamp to the
# pool are both reachable at small k, where the equal share overshoots a small pool.
POOL_SIZES = {"A": 100, "B": 60, "C": 40, "D": 17, "rare": 3}
# The same shape with a spare cluster, so the OTHER role can be a cluster of its own: a
# cluster used as both source and sink can draw the same cell twice and unbalance the flow.
CLAMP_SIZES = {"A": 100, "B": 60, "C": 40, "D": 17, "rare": 3, "snk": 25}
# Pools far enough apart that neither the floor nor the clamp can bind at any k: the equal
# share is 20/3 = 6.67 and the proportional one at most 10, all inside [1, pool]. Three
# clusters per role, with the roles disjoint.
FREE_SIZES = {"A": 100, "B": 60, "C": 40, "P": 100, "Q": 60, "R": 40}
# One pool so small that round(p_end * pool) is 0 and the floor of 1 lifts it. That raise is
# already inside budget_role, so the split hands its weight-share to 'big' and floors it a
# second time: the realized total comes out ABOVE the budget. The other drift direction.
DRIFT_SIZES = {"big": 20, "tiny": 1, "snk": 10}
# Axolotl telencephalon clusters, with their real names: the tie-break is alphabetical, so the
# names, not just the sizes, decide who takes the leftover unit at k=0.
AXOLOTL_SIZES = {"reaEGC": 333, "wntEGC": 232, "nptxEX": 177, "mpEX": 554, "dpEX": 76}


@pytest.fixture(scope="module")
def pool_object():
    return _pool_object(POOL_SIZES)


@pytest.fixture(scope="module")
def clamp_object():
    return _pool_object(CLAMP_SIZES)


@pytest.fixture(scope="module")
def free_object():
    return _pool_object(FREE_SIZES)


@pytest.fixture(scope="module")
def drift_object():
    return _pool_object(DRIFT_SIZES)


@pytest.fixture(scope="module")
def axolotl_object():
    return _pool_object(AXOLOTL_SIZES)


@pytest.fixture(scope="module")
def toy_graph():
    """The toy system's integrated graph. Clusters source/mid/sink, 50 cells each."""
    return dt.pp.integrate(*line_toy(n=150), verbose=0)


def _quotas(obj, source_clusters, sink_clusters, **kw):
    """The per-cluster quotas the sampler actually used, read off the level-2 log.

    Parameters
    ----------
    obj : anndata.AnnData
        Object carrying the graph and the cluster labels.
    source_clusters, sink_clusters : list of str, or None
        The two roles, as for :func:`deltatrace.tl.endpoints`.
    **kw
        Forwarded to :func:`deltatrace.tl.endpoints`.

    Returns
    -------
    tuple of (dict, dict, dict)
        Source quotas, sink quotas, and the endpoints result.
    """
    kw.setdefault("sampling_mode", "random")
    kw.setdefault("p_basis", "cluster")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gr = dt.tl.endpoints(obj, source_clusters, sink_clusters,
                             cluster_key="clusters", graph="intr", verbose=2, **kw)
    out = []
    for want in ("Source quotas:", "Sink quotas:"):
        lines = [ln for ln in buf.getvalue().splitlines() if want in ln]
        assert len(lines) == 1, buf.getvalue()
        out.append(ast.literal_eval(lines[0].split(want)[1].strip()))
    return out[0], out[1], gr


def _counts(idxs, sizes):
    """Sampled cells per cluster, from the returned indices and the layout order."""
    bounds, at = {}, 0
    for c, n in sizes.items():
        bounds[c] = range(at, at + n)
        at += n
    got = {}
    for i in idxs:
        for c, rng in bounds.items():
            if int(i) in rng:
                got[c] = got.get(c, 0) + 1
    return got


# --------------------------------------------------------------------------------------
# k = 1 is the pre-change code, index for index
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode,table", [("random", K1_RANDOM), ("directed", K1_DIRECTED)])
def test_alloc_power_one_is_the_historical_cluster_allocation(pool_object, mode, table):
    """alloc_power=1.0 asks for what 'cluster' asked for before it existed, and pins the cells.

    The quotas are the pre-change claim, index for index is a drift guard on the current
    sampler. Two source clusters and three sink clusters, so both quota calls are exercised.
    """
    for seed in SEEDS:
        want_src, want_snk = table[seed]
        for k in (1.0, 1):  # a float and the int, which must be accepted and identical
            gr = dt.tl.endpoints(pool_object, ["A", "B"], ["C", "D", "rare"],
                                 cluster_key="clusters", graph="intr", p_end=0.10,
                                 sampling_mode=mode, seed=seed, verbose=0,
                                 p_basis="cluster", alloc_power=k)
            assert [int(i) for i in gr["source_indices"]] == want_src, f"seed={seed}, k={k!r}"
            assert [int(i) for i in gr["sink_indices"]] == want_snk, f"seed={seed}, k={k!r}"
            assert (_counts(gr["source_indices"], POOL_SIZES),
                    _counts(gr["sink_indices"], POOL_SIZES)) == K1_QUOTAS, f"quota drift, k={k!r}"
        # Both roles really are multi-cluster, or the pinning above is weaker than it looks.
        assert len(_counts(want_src, POOL_SIZES)) == 2
        assert len(_counts(want_snk, POOL_SIZES)) == 3


def test_default_alloc_power_is_proportional(pool_object):
    """Omitting alloc_power must be k=1, the historical allocation, not a new one."""
    for seed in SEEDS:
        common = dict(cluster_key="clusters", graph="intr", p_end=0.10,
                      sampling_mode="directed", seed=seed, verbose=0, p_basis="cluster")
        a = dt.tl.endpoints(pool_object, ["A", "B"], ["C", "D", "rare"],
                            **common)
        b = dt.tl.endpoints(pool_object, ["A", "B"], ["C", "D", "rare"],
                            alloc_power=1.0, **common)
        assert list(a["source_indices"]) == list(b["source_indices"])
        assert list(a["sink_indices"]) == list(b["sink_indices"])


# --------------------------------------------------------------------------------------
# what k in between actually produces
# --------------------------------------------------------------------------------------
def test_alloc_power_half_reproduces_the_axolotl_numbers(axolotl_object):
    """The square-root allocation on these axolotl sinks: 177/554/76 at p_end=0.15 -> 35/63/23.

    The budget is the k=1 total, 27 + 83 + 11 = 121, and the square-root split spends all of
    it: 35 + 63 + 23 = 121. Written as literals rather than recomputed, so a change to the
    weighting, the budget or the remainder rule has to move a number here.
    """
    snk = ["nptxEX", "mpEX", "dpEX"]
    _, q1, _ = _quotas(axolotl_object, ["reaEGC", "wntEGC"], snk, p_end=0.15,
                       seed=42, alloc_power=1.0)
    assert q1 == {"nptxEX": 27, "mpEX": 83, "dpEX": 11}
    assert sum(q1.values()) == 121
    _, qh, gr = _quotas(axolotl_object, ["reaEGC", "wntEGC"], snk, p_end=0.15,
                        seed=42, alloc_power=0.5)
    assert qh == {"nptxEX": 35, "mpEX": 63, "dpEX": 23}
    assert sum(qh.values()) == 121, "the square-root split must spend the k=1 budget"
    assert _counts(gr["sink_indices"], AXOLOTL_SIZES) == qh, "quota not honored by the draw"
    # Between the ends in the direction sampling theory says: less spread than proportional.
    assert max(q1.values()) / min(q1.values()) > max(qh.values()) / min(qh.values()) > 1.0


def test_alloc_power_zero_is_equal_up_to_the_remainder(axolotl_object):
    """k=0 ignores pool size: 121 over three sinks is 40 each, and the leftover goes by name.

    121 / 3 = 40.333, so every cluster floors to 40 and one unit is left. All three
    fractional parts are exactly equal, which is the case the tie-break exists for: it goes
    to the alphabetically first name, 'dpEX', and not to the largest pool ('mpEX') or the
    first cluster listed ('nptxEX').
    """
    snk = ["nptxEX", "mpEX", "dpEX"]
    _, q0, _ = _quotas(axolotl_object, ["reaEGC"], snk, p_end=0.15, seed=42,
                       alloc_power=0.0)
    assert q0 == {"nptxEX": 40, "mpEX": 40, "dpEX": 41}
    assert sum(q0.values()) == 121
    assert max(q0.values()) - min(q0.values()) == 1, "an equal split is equal within one"
    assert min(sorted(snk)) == "dpEX", "the fixture no longer tests the name tie-break"


def test_total_is_preserved_when_no_floor_or_clamp_binds(free_object):
    """sum(q) == the k=1 budget at every k, when neither guard can bind.

    Pools 100/60/40 per role at p_end=0.10: the budget is 10 + 6 + 4 = 20, the equal share is
    6.67 and the proportional maximum is 10, so every quota stays inside [1, pool] at every k
    and the largest-remainder reconciliation is the only thing deciding the integers. The two
    roles are disjoint clusters, so nothing is drawn twice.
    """
    for k in KS:
        qs, qk, gr = _quotas(free_object, ["A", "B", "C"], ["P", "Q", "R"],
                             p_end=0.10, seed=42, alloc_power=k)
        assert sum(qs.values()) == 20, f"source total drifted at k={k}: {qs}"
        assert sum(qk.values()) == 20, f"sink total drifted at k={k}: {qk}"
        for q in (qs, qk):
            for c, v in q.items():
                assert 1 < v < FREE_SIZES[c], f"a guard bound at k={k}: the claim is vacuous"
        assert len(gr["source_indices"]) == 20
        assert len(gr["sink_indices"]) == 20


def test_floor_and_clamp_may_pull_the_total_below_the_budget(clamp_object):
    """The documented drift: at k=0 a 3-cell cluster is asked for 4 and can only give 3.

    Budget 23 at p_end=0.10 (10+6+4+2+1), equal shares 4.6 each. 'rare' has 3 cells, so its
    share is clamped and the realized total is 22, not 23. No second pass hands those cells
    to anyone else -- re-balancing would have to iterate, and the iteration has no fixed
    point once every cluster sits at its clamp.
    """
    qs, _, gr = _quotas(clamp_object, list(POOL_SIZES), ["snk"], p_end=0.10, seed=42,
                        alloc_power=0.0)
    assert qs == {"A": 5, "B": 5, "C": 5, "D": 4, "rare": 3}
    assert qs["rare"] == POOL_SIZES["rare"], "the clamp is what this test is about"
    assert sum(qs.values()) == 22
    assert len(gr["source_indices"]) == 22


def test_floor_may_push_the_total_above_the_budget(drift_object):
    """The other drift direction, which the floor of 1 causes and which is not confined to k=0.

    Pools big=20 and tiny=1 at p_end=0.05. round(0.05 * 1) is 0, so budget_role already carries
    tiny's raise-to-1: 1 + 1 = 2. At k=0.5 the split gives that weight-share to 'big', which
    lands on 2, and tiny is floored to 1 a second time -- realized total 3 against a budget of
    2. So budget_role is not an upper bound, and the drift is two-sided.
    """
    qs, _, gr = _quotas(drift_object, ["big", "tiny"], ["snk"], p_end=0.05, seed=42,
                        alloc_power=0.5)
    budget = 1 + 1
    assert qs == {"big": 2, "tiny": 1}
    assert sum(qs.values()) == 3 > budget
    assert len(gr["source_indices"]) == 3


# --------------------------------------------------------------------------------------
# the two invariances the parameter exists for
# --------------------------------------------------------------------------------------
def test_quotas_are_seed_independent_at_every_alloc_power(pool_object):
    """One quota vector per k across every seed, even though the sample within a cluster moves.

    Largest remainder, not a shuffle: this is the property ``p_basis="lcc"`` lacks, and the
    reason a sweep over k is readable at all.
    """
    for k in KS:
        quotas, samples = set(), set()
        for seed in SEEDS:
            qs, qk, gr = _quotas(pool_object, ["A", "B"], ["C", "D", "rare"],
                                 p_end=0.10, seed=seed, alloc_power=k)
            quotas.add((tuple(sorted(qs.items())), tuple(sorted(qk.items()))))
            samples.add(tuple(sorted(int(i) for i in gr["source_indices"])))
        assert len(quotas) == 1, f"quotas moved with the seed at k={k}: {quotas}"
        assert len(samples) > 1, "the fixture no longer varies the sample: claim is vacuous"


def test_quotas_are_invariant_to_the_order_of_the_cluster_list(pool_object):
    """Reversing a role's cluster list must not move a single quota, at any k.

    Ties in the largest-remainder step break on the cluster name, so the allocation cannot
    depend on how the caller happened to write the list. Reversal is the sharpest probe:
    dict insertion order, and therefore the order the leftover units would be handed out in
    if the tie-break were positional, is exactly inverted.
    """
    for k in KS:
        qs, qk, _ = _quotas(pool_object, ["A", "B"], ["C", "D", "rare"],
                            p_end=0.10, seed=42, alloc_power=k)
        qs_r, qk_r, _ = _quotas(pool_object, ["B", "A"], ["rare", "D", "C"],
                                p_end=0.10, seed=42, alloc_power=k)
        assert qs_r == qs, f"source quotas depend on list order at k={k}"
        assert qk_r == qk, f"sink quotas depend on list order at k={k}"


def test_largest_and_smallest_move_monotonically_with_alloc_power(clamp_object):
    """Raising k moves cells from the small clusters to the large ones on this pool set.

    The largest pool's quota is non-decreasing in k here and the smallest pool's
    non-increasing, which is the direction the parameter exists for: k=0 protects small
    strata, k=1 follows the population. It is a TENDENCY of the integer quotas, not a
    per-step guarantee, and this test pins it only on this pool set and this k grid. Two
    things can hand a unit back elsewhere: the largest-remainder ranks reshuffle as k moves
    inside k<1, and k=1 is a separate rounding rule (the bit-identity short circuit) rather
    than the limit of the family, so q at k=1 can differ from q just below it. Exact
    monotonicity holds for the real-valued shares, and empirically at every k on the pool
    shapes tried. Also asserted to actually MOVE, so a stub returning one
    allocation for every k could not pass.
    """
    big, small = "A", "rare"
    got = []
    for k in KS:
        qs, _, _ = _quotas(clamp_object, list(POOL_SIZES), ["snk"], p_end=0.10,
                           seed=42, alloc_power=k)
        got.append((k, qs[big], qs[small]))
    for (k0, big0, small0), (k1, big1, small1) in itertools.pairwise(got):
        assert big1 >= big0, f"largest cluster fell from k={k0} to k={k1}: {got}"
        assert small1 <= small0, f"smallest cluster rose from k={k0} to k={k1}: {got}"
    assert got[-1][1] > got[0][1], f"k did nothing to the largest cluster: {got}"
    assert got[-1][2] < got[0][2], f"k did nothing to the smallest cluster: {got}"


# --------------------------------------------------------------------------------------
# validation, inertness, forwarding
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-0.1, 1.0001, 2, 10.0, -1, float("nan"), float("inf"),
                                 float("-inf")])
def test_alloc_power_outside_the_unit_range_raises(pool_object, bad):
    """Only [0, 1] is meaningful; the message names the range and what its ends mean."""
    with pytest.raises(ValueError) as e:
        dt.tl.endpoints(pool_object, ["A", "B"], ["C"], cluster_key="clusters",
                        graph="intr", p_end=0.10, verbose=0, p_basis="cluster",
                        alloc_power=bad)
    msg = str(e.value)
    assert "alloc_power" in msg
    assert "[0, 1]" in msg, msg
    assert "equal" in msg and "proportional" in msg, msg


@pytest.mark.parametrize("bad", ["0.5", "half", None, [0.5], {"A": 0.5}, True, False])
def test_alloc_power_non_numeric_raises(pool_object, bad):
    """A non-number is a caller error, bools included: True is not an exponent."""
    with pytest.raises(ValueError) as e:
        dt.tl.endpoints(pool_object, ["A", "B"], ["C"], cluster_key="clusters",
                        graph="intr", p_end=0.10, verbose=0, p_basis="cluster",
                        alloc_power=bad)
    msg = str(e.value)
    assert "alloc_power" in msg
    assert repr(bad) in msg, msg


def test_bad_alloc_power_is_rejected_before_any_work(pool_object, monkeypatch):
    """The check runs up front: a bad exponent must not cost the caller a graph build."""
    from deltatrace import _endpoints

    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("alloc_power validated too late: the graph was already built")

    monkeypatch.setattr(_endpoints, "_build_cell_graph", boom)
    with pytest.raises(ValueError, match="alloc_power"):
        dt.tl.endpoints(pool_object, ["A"], ["C"], cluster_key="clusters",
                        graph="intr", verbose=0, p_basis="cluster", alloc_power=3.0)
    assert not calls

    # Without this the test passes vacuously: the patched builder must be the one
    # endpoints actually calls, so a valid alloc_power has to run straight into it.
    with pytest.raises(AssertionError, match="graph was already built"):
        dt.tl.endpoints(pool_object, ["A"], ["C"], cluster_key="clusters",
                        graph="intr", verbose=0, p_basis="cluster", alloc_power=0.5)
    assert len(calls) == 1


def test_alloc_power_is_inert_under_the_lcc_basis(toy_graph):
    """'lcc' has no per-pool weights to raise to a power, so k must change nothing.

    Not an error either: like ``directed_metric`` under a mode that never centers anything,
    a valid value under a basis that cannot use it is silently inert.
    """
    for seed in (0, 42, 2024):
        common = dict(cluster_key="clusters", graph="intr", p_end=0.05,
                      sampling_mode="directed", seed=seed, verbose=0, p_basis="lcc")
        base = dt.tl.endpoints(toy_graph, ["source", "mid"], ["mid", "sink"],
                               **common)
        for k in (0.0, 0.25, 0.5, 1.0):
            got = dt.tl.endpoints(toy_graph, ["source", "mid"], ["mid", "sink"],
                                  alloc_power=k, **common)
            assert list(got["source_indices"]) == list(base["source_indices"]), f"k={k}"
            assert list(got["sink_indices"]) == list(base["sink_indices"]), f"k={k}"


def test_trajectory_forwards_alloc_power(pool_object):
    """trajectory must hand alloc_power to endpoints, and it must reach the pseudotime."""
    obj_p, obj_e = pool_object.copy(), pool_object.copy()
    common = dict(cluster_key="clusters", graph="intr", basis="spatial",
                  reconnect_graph="intr", p_end=0.10, seed=42,
                  sampling_mode="random", p_basis="cluster", verbose=0)
    res_p = dt.tl.trajectory(obj_p, ["A", "B"], ["C", "D", "rare"], alloc_power=1.0,
                             **common)
    res_e = dt.tl.trajectory(obj_e, ["A", "B"], ["C", "D", "rare"], alloc_power=0.0,
                             **common)
    # k=1 follows the pools: A 10, B 6 and C 4, D 2, rare 1. k=0 ignores them: 8/8 and
    # 3/2/2, the leftover unit going to 'C' on the name tie-break. Same 16 and 7 totals.
    assert _counts(res_p["endpoints"]["source_indices"], POOL_SIZES) == {"A": 10, "B": 6}
    assert _counts(res_e["endpoints"]["source_indices"], POOL_SIZES) == {"A": 8, "B": 8}
    assert _counts(res_p["endpoints"]["sink_indices"], POOL_SIZES) == {"C": 4, "D": 2, "rare": 1}
    assert _counts(res_e["endpoints"]["sink_indices"], POOL_SIZES) == {"C": 3, "D": 2, "rare": 2}
    assert not np.allclose(obj_p.obs["pseudotime"].values, obj_e.obs["pseudotime"].values)


def test_trajectory_defaults_to_proportional(pool_object):
    """trajectory's default must match endpoints' default, not diverge from it."""
    obj_a, obj_b = pool_object.copy(), pool_object.copy()
    common = dict(cluster_key="clusters", graph="intr", basis="spatial",
                  reconnect_graph="intr", p_end=0.10, seed=42,
                  sampling_mode="random", p_basis="cluster", verbose=0)
    a = dt.tl.trajectory(obj_a, ["A", "B"], ["C", "D", "rare"], **common)
    b = dt.tl.trajectory(obj_b, ["A", "B"], ["C", "D", "rare"], alloc_power=1.0, **common)
    assert list(a["endpoints"]["source_indices"]) == list(b["endpoints"]["source_indices"])
    assert list(a["endpoints"]["sink_indices"]) == list(b["endpoints"]["sink_indices"])
