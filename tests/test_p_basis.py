"""What ``p_end`` is a fraction OF, and what each ``p_basis`` pins.

Two bases, and the whole point of the option is that they answer different questions.

``p_basis="lcc"`` is the historical sizing: one total per role, ``round(p_end * n_lcc)``,
split equally across that role's clusters with the remainder handed to a seeded random
draw. Cluster size is ignored and the per-cluster quotas move with ``seed``. Those QUOTAS
are frozen here against the pre-change numbers, because it was the default basis before
0.1.0. The sampled indices are pinned separately: they were re-captured
when the sampler moved to a per-(role, cluster) stream, so they guard against drift from
here on rather than certify agreement with the tree before ``p_basis``.

``p_basis="cluster"`` sizes each cluster from its own eligibility pool for that role,
``max(1, round(p_end * n_eligible_c))`` clamped to the pool. The quotas are then a function
of the arguments and the data alone: no shared total, no remainder, no draw, so no ``seed``
in the quota. The price is that the source and sink totals no longer match, which the tests
below assert rather than tolerate. It is the default: every declared cluster gets at
least one cell, where ``"lcc"`` can hand a cluster none.
"""

import ast
import contextlib
import io
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from _line_toy import line_toy
from scipy.sparse.csgraph import connected_components

import deltatrace as dt

SEEDS = (0, 1, 7, 42, 123, 2024)

# What "lcc" is frozen against. _split_quota's remainder now draws from a stream keyed on
# ("quota", role), independent of the OTHER role and of both lists' order and length, so
# these numbers are a property of (seed, role, the role's own cluster set) alone. Note the
# sink split still moves with the seed (3+4 cells at seed 0, 4+3 at seed 7) -- that is the
# remainder draw, and pinning it here is the point.
QUOTAS = {
    # seed: (source counts, sink counts), toy(n=150), ["source"] -> ["mid", "sink"]
    0: ({"source": 7}, {"mid": 3, "sink": 4}),
    1: ({"source": 7}, {"mid": 3, "sink": 4}),
    7: ({"source": 7}, {"mid": 4, "sink": 3}),
    42: ({"source": 7}, {"mid": 3, "sink": 4}),
    123: ({"source": 7}, {"mid": 3, "sink": 4}),
    2024: ({"source": 7}, {"mid": 3, "sink": 4}),
}
SAMPLED = {
    # seed: (source_indices, sink_indices), toy(n=150), ["source"] -> ["mid", "sink"],
    #       p_end=0.05, sampling_mode="directed"
    0: ([25, 21, 29, 20, 17, 22, 18], [92, 99, 95, 146, 148, 147, 145]),
    1: ([25, 17, 22, 28, 24, 32, 23], [92, 99, 95, 146, 148, 147, 145]),
    7: ([25, 34, 32, 16, 20, 30, 27], [92, 99, 95, 93, 146, 148, 147]),
    42: ([25, 29, 31, 18, 19, 20, 24], [92, 99, 95, 146, 148, 147, 145]),
    123: ([25, 21, 22, 23, 20, 30, 34], [92, 99, 95, 146, 148, 147, 145]),
    2024: ([25, 22, 23, 17, 16, 26, 29], [92, 99, 95, 146, 148, 147, 145]),
}
# Same tree, ["source", "mid"] -> ["sink"], p_end=0.05, sampling_mode="random".
QUOTAS_RANDOM = {
    0: ({"source": 3, "mid": 4}, {"sink": 7}),
    1: ({"source": 3, "mid": 4}, {"sink": 7}),
    7: ({"source": 4, "mid": 3}, {"sink": 7}),
    42: ({"source": 4, "mid": 3}, {"sink": 7}),
    123: ({"source": 3, "mid": 4}, {"sink": 7}),
    2024: ({"source": 3, "mid": 4}, {"sink": 7}),
}
SAMPLED_RANDOM = {
    0: ([14, 45, 28, 78, 89, 80, 50], [126, 146, 105, 107, 100, 128, 122]),
    1: ([37, 5, 19, 63, 74, 50, 68], [103, 133, 143, 149, 124, 140, 127]),
    7: ([6, 24, 30, 25, 77, 90, 82], [130, 109, 119, 117, 146, 105, 133]),
    42: ([23, 37, 27, 3, 73, 81, 54], [101, 130, 147, 104, 141, 122, 110]),
    123: ([17, 46, 18, 67, 94, 65, 98], [117, 145, 134, 123, 100, 115, 140]),
    2024: ([41, 20, 29, 93, 86, 79, 55], [139, 112, 125, 108, 135, 149, 110]),
}
# Same tree, ["source", "mid"] -> ["mid", "sink"], p_end=0.05, "directed".
#
# Two clusters on BOTH roles, which the two configs above do not have. The source and sink
# remainder draws now come off independent streams keyed on role, so the order the two
# _split_quota calls happen to run in was never observable to begin with; what this config
# adds is a SECOND remainder draw to pin (the other two only exercise one, since the other
# role has a single cluster and nothing to shuffle).
QUOTAS_BOTH_MULTI = {
    0: ({"source": 3, "mid": 4}, {"mid": 3, "sink": 4}),
    1: ({"source": 3, "mid": 4}, {"mid": 3, "sink": 4}),
    7: ({"source": 4, "mid": 3}, {"mid": 4, "sink": 3}),
    42: ({"source": 4, "mid": 3}, {"mid": 3, "sink": 4}),
    123: ({"source": 3, "mid": 4}, {"mid": 3, "sink": 4}),
    2024: ({"source": 3, "mid": 4}, {"mid": 3, "sink": 4}),
}
SAMPLED_BOTH_MULTI = {
    0: ([25, 21, 29, 70, 67, 68, 77], [92, 99, 95, 146, 148, 147, 145]),
    1: ([25, 17, 22, 70, 72, 78, 73], [92, 99, 95, 146, 148, 147, 145]),
    7: ([25, 34, 32, 16, 70, 68, 71], [92, 99, 95, 93, 146, 148, 147]),
    42: ([25, 29, 31, 18, 70, 63, 65], [92, 99, 95, 146, 148, 147, 145]),
    123: ([25, 21, 22, 70, 74, 69, 65], [92, 99, 95, 146, 148, 147, 145]),
    2024: ([25, 22, 23, 70, 65, 68, 69], [92, 99, 95, 146, 148, 147, 145]),
}


TOY_SIZES = {"source": 50, "mid": 50, "sink": 50}   # cell layout of toy(n=150), in order


@pytest.fixture(scope="module")
def toy_graph():
    """The toy system's integrated graph. Clusters source/mid/sink, 50 cells each.

    Built at the pre-0.1.0 graph defaults (alpha=0.5, p_keep=0.1), spelled out: the quotas
    and indices pinned below were recorded on this graph, and a change of pp.integrate'
    defaults must not move them."""
    return dt.pp.integrate(*line_toy(n=150), alpha=0.5, p_keep=0.1, verbose=0)


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


# One tiny cluster ("rare"), so the floor of 1 and the clamp to the pool are both reachable.
POOL_SIZES = {"A": 100, "B": 60, "C": 40, "rare": 3}


@pytest.fixture(scope="module")
def pool_object():
    return _pool_object(POOL_SIZES)


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
    kw.setdefault("sampling_mode", "directed")
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


def _counts(gr, sizes):
    """Sampled source and sink cells per cluster, from the returned indices."""
    bounds, at = {}, 0
    for c, n in sizes.items():
        bounds[c] = range(at, at + n)
        at += n
    def per(idxs):
        got = {}
        for i in idxs:
            for c, rng in bounds.items():
                if i in rng:
                    got[c] = got.get(c, 0) + 1
        return got
    return per(gr["source_indices"]), per(gr["sink_indices"])


# --------------------------------------------------------------------------------------
# "lcc": frozen, including the parts that are not ideal
# --------------------------------------------------------------------------------------
def test_lcc_basis_quotas_are_pinned(toy_graph):
    """'lcc' splits equally with a seed-dependent remainder, frozen here, both roles.

    Quotas, not indices: the equal split and its seed-dependent remainder draw are what this
    basis is frozen for. The indices come off a per-(role, cluster) stream and are pinned
    separately, below.
    """
    for seed, (want_src, want_snk) in QUOTAS.items():
        gr = dt.tl.endpoints(toy_graph, ["source"], ["mid", "sink"],
                             cluster_key="clusters", graph="intr", p_end=0.05,
                             sampling_mode="directed", seed=seed, verbose=0,
                             p_basis="lcc")
        got_s, got_k = _counts(gr, TOY_SIZES)
        assert got_s == want_src, f"source quota drift, seed={seed}"
        assert got_k == want_snk, f"sink quota drift, seed={seed}"


def test_lcc_basis_sampled_indices_are_pinned(toy_graph):
    """The cells themselves, index for index, so nothing moves them unnoticed."""
    for seed, (want_src, want_snk) in SAMPLED.items():
        gr = dt.tl.endpoints(toy_graph, ["source"], ["mid", "sink"],
                             cluster_key="clusters", graph="intr", p_end=0.05,
                             sampling_mode="directed", seed=seed, verbose=0,
                             p_basis="lcc")
        assert [int(i) for i in gr["source_indices"]] == want_src, f"source drift, seed={seed}"
        assert [int(i) for i in gr["sink_indices"]] == want_snk, f"sink drift, seed={seed}"


def test_lcc_basis_random_mode_quotas_and_indices(toy_graph):
    """Same, with two source clusters and the plain random sampler."""
    for seed in SEEDS:
        gr = dt.tl.endpoints(toy_graph, ["source", "mid"], ["sink"],
                             cluster_key="clusters", graph="intr", p_end=0.05,
                             sampling_mode="random", seed=seed, verbose=0,
                             p_basis="lcc")
        assert _counts(gr, TOY_SIZES) == QUOTAS_RANDOM[seed], f"quota drift, seed={seed}"
        want_src, want_snk = SAMPLED_RANDOM[seed]
        assert [int(i) for i in gr["source_indices"]] == want_src, f"source drift, seed={seed}"
        assert [int(i) for i in gr["sink_indices"]] == want_snk, f"sink drift, seed={seed}"


def test_lcc_basis_both_roles_multi_quotas_and_indices(toy_graph):
    """Same, with 2+ clusters on BOTH roles, so both remainder draws are exercised.

    The other two configs each have a single cluster on one role, and shuffling a 1-element
    list draws nothing, so only one of the two ``_split_quota`` calls actually consumes its
    stream. Here both do, pinning the source and sink remainder draws together.
    """
    for seed in SEEDS:
        gr = dt.tl.endpoints(toy_graph, ["source", "mid"], ["mid", "sink"],
                             cluster_key="clusters", graph="intr", p_end=0.05,
                             sampling_mode="directed", seed=seed, verbose=0,
                             p_basis="lcc")
        assert _counts(gr, TOY_SIZES) == QUOTAS_BOTH_MULTI[seed], f"quota drift, seed={seed}"
        want_src, want_snk = SAMPLED_BOTH_MULTI[seed]
        assert [int(i) for i in gr["source_indices"]] == want_src, f"source drift, seed={seed}"
        assert [int(i) for i in gr["sink_indices"]] == want_snk, f"sink drift, seed={seed}"


def test_cluster_is_the_default(toy_graph):
    """Omitting p_basis must be the benchmarked basis, "cluster"."""
    for seed in SEEDS:
        common = dict(cluster_key="clusters", graph="intr", p_end=0.05,
                      sampling_mode="directed", seed=seed, verbose=0)
        a = dt.tl.endpoints(toy_graph, ["source"], ["mid", "sink"], **common)
        b = dt.tl.endpoints(toy_graph, ["source"], ["mid", "sink"],
                            p_basis="cluster", **common)
        assert list(a["source_indices"]) == list(b["source_indices"])
        assert list(a["sink_indices"]) == list(b["sink_indices"])


def test_lcc_basis_quotas_are_seed_dependent(pool_object):
    """The equal split's remainder draw is preserved: which cluster gets the extra moves.

    This is the behavior 'cluster' exists to avoid. 'lcc' was the default before 0.1.0 and
    is still supported, so a later attempt to quietly de-randomize it has to argue with a
    test. At p_end=0.01 one source cluster gets no cells, which has to be said, and a
    source-only warning must not talk about lineages.
    """
    seen = set()
    for seed in SEEDS:
        with pytest.warns(UserWarning, match=r"get none\. Raise p_end") as rec:
            qs, _, _ = _quotas(pool_object, ["A", "B", "C"], ["rare"], p_end=0.01,
                               seed=seed, sampling_mode="random", p_basis="lcc")
        assert not any("lineages" in str(w.message) for w in rec)
        seen.add(tuple(qs[c] for c in ("A", "B", "C")))
    assert len(seen) > 1, f"remainder no longer depends on the seed: {seen}"
    # Cluster size is ignored: every quota is within one of every other.
    for q in seen:
        assert max(q) - min(q) <= 1, q


# --------------------------------------------------------------------------------------
# "cluster": proportional, floored, clamped, seed-free
# --------------------------------------------------------------------------------------
def test_cluster_basis_quota_is_proportional_to_each_pool(pool_object):
    """q_c = round(p_end * pool_c), per cluster, and the role totals are free to differ."""
    p_end = 0.10
    qs, qk, gr = _quotas(pool_object, ["A", "B"], ["C", "rare"], p_end=p_end,
                         seed=42, p_basis="cluster")
    assert qs == {"A": 10, "B": 6}
    assert qk == {"C": 4, "rare": 1}
    got_s, got_k = _counts(gr, POOL_SIZES)
    assert got_s == qs and got_k == qk
    # The equal-totals property of "lcc" is gone, by design.
    assert sum(qs.values()) != sum(qk.values())
    # ...and the flow is still balanced, because f normalizes per role.
    assert abs(float(np.sum(gr["f"]))) < 1e-12


def test_cluster_basis_floors_tiny_cluster_at_one(pool_object):
    """A cluster smaller than 1/p_end still contributes one cell, not zero.

    3 cells at p_end=0.10 rounds to 0. The floor makes it 1, so a rare terminal is
    represented, but at 33 percent of itself rather than 10: the floor is representation,
    not proportionality.
    """
    _, qk, gr = _quotas(pool_object, ["A"], ["rare"], p_end=0.10, seed=42,
                        p_basis="cluster")
    assert int(np.round(0.10 * POOL_SIZES["rare"])) == 0, "fixture no longer tests the floor"
    assert qk == {"rare": 1}
    assert len(gr["sink_indices"]) == 1
    # Under "lcc" the same cluster is asked for the whole role total and cannot supply
    # it, so it is clamped to everything it has rather than refusing.
    with pytest.warns(UserWarning, match="sampling all 3"):
        gr_lcc = dt.tl.endpoints(pool_object, ["A"], ["rare"],
                                 cluster_key="clusters", graph="intr", p_end=0.10, seed=42,
                                 verbose=0, p_basis="lcc")
    assert len(gr_lcc["sink_indices"]) == 3


def test_cluster_basis_quotas_are_seed_independent(pool_object):
    """One quota vector across every seed, even though the sample within a cluster moves."""
    quotas, samples = set(), set()
    for seed in SEEDS:
        qs, qk, gr = _quotas(pool_object, ["A", "B"], ["C", "rare"], p_end=0.10,
                             seed=seed, sampling_mode="random", p_basis="cluster")
        quotas.add((tuple(sorted(qs.items())), tuple(sorted(qk.items()))))
        samples.add(tuple(sorted(map(int, gr["source_indices"]))))
    assert len(quotas) == 1, f"quotas moved with the seed: {quotas}"
    assert len(samples) > 1, "fixture no longer varies the sample, so the claim is vacuous"


def test_cluster_basis_dual_role_gets_two_independent_quotas(pool_object):
    """A cluster named as both source and sink is sized twice, once per role pool.

    Its two quotas are equal here only because the two pools are the same cells; what the
    test pins is that each role is sized from its OWN pool, so B's source quota tracks the
    source list and its sink quota the sink list, independently of the other role's mix.
    """
    qs, qk, _ = _quotas(pool_object, ["A", "B"], ["B", "C"], p_end=0.10, seed=42,
                        p_basis="cluster")
    assert "B" in qs and "B" in qk
    assert qs == {"A": 10, "B": 6}
    assert qk == {"B": 6, "C": 4}
    # Adding a cluster to one role must not touch the other role's quotas: no shared total.
    qs2, qk2, _ = _quotas(pool_object, ["A", "B"], ["B", "C", "rare"], p_end=0.10,
                          seed=42, p_basis="cluster")
    assert qs2 == qs
    assert qk2 == {"B": 6, "C": 4, "rare": 1}


def test_cluster_basis_never_requests_more_than_the_pool(pool_object):
    """The quota is clamped to the eligible pool, so an over-unity p_end cannot overdraw."""
    for p_end in (1.0, 2.0, 7.5):
        qs, qk, gr = _quotas(pool_object, ["rare"], ["C"], p_end=p_end, seed=42,
                             sampling_mode="random", p_basis="cluster")
        assert qs == {"rare": POOL_SIZES["rare"]}
        assert qk == {"C": POOL_SIZES["C"]}
        assert len(gr["source_indices"]) == POOL_SIZES["rare"]
        assert len(gr["sink_indices"]) == POOL_SIZES["C"]


def test_cluster_basis_pool_is_per_role_and_lcc_restricted(toy_graph):
    """The pool is cluster INTERSECT LCC, not the raw cluster size.

    All three toy clusters hold exactly 50 cells, but two "sink" cells sit off the LCC, so
    the sink pool is 48 against the source's 50. ``p_end=0.30`` is chosen because
    it separates the two: ``round(0.30 * 50) = 15`` and ``round(0.30 * 48) = 14``. The
    expected quotas are written as literals rather than recomputed from the pool, so a
    sizing that reached for the raw cluster size would have to produce 15 sinks and be
    caught. At ``p_end=0.10`` both give 5 and the assertion says nothing.

    The off-LCC membership is computed here from the neighbor graph with scipy, not read
    back out of the package, so the pool the test believes in is derived independently of
    the code that sizes it.
    """
    labels = toy_graph.obs["clusters"].astype(str).values
    n_comp, comp = connected_components(toy_graph.obsp["neighbors_intr_distances"],
                                        directed=True, connection="weak")
    lcc = int(np.argmax(np.bincount(comp)))
    off = np.flatnonzero(comp != lcc)
    assert n_comp > 1 and len(off) > 0, "fixture no longer has off-LCC cells"
    # Raw sizes are equal, so a quota of 14 can only have come from an LCC-restricted pool.
    assert int((labels == "source").sum()) == int((labels == "sink").sum()) == 50
    assert int(((labels == "source") & (comp == lcc)).sum()) == 50
    assert int(((labels == "sink") & (comp == lcc)).sum()) == 48

    qs, qk, gr = _quotas(toy_graph, ["source"], ["sink"], p_end=0.30, seed=42,
                         sampling_mode="random", p_basis="cluster")
    assert qs == {"source": 15}
    assert qk == {"sink": 14}, "sink sized off the raw cluster size, not the LCC pool"
    assert len(gr["sink_indices"]) == 14
    # Both off-LCC cells are sinks, so the sink role is the only one where this is visible.
    assert set(labels[off]) == {"sink"}
    assert set(int(i) for i in gr["sink_indices"]).isdisjoint(off.tolist())


def test_cluster_basis_rounds_half_to_even(pool_object):
    """The documented rounding rule is NumPy's half-to-even, and only that rule fits.

    At ``p_end=0.025`` the products are exactly 2.5 for A (100 cells) and exactly
    1.5 for B (60), and half-to-even sends both to the even value 2 -- 2.5 DOWN and 1.5 UP.
    No other rule produces that pair: round-half-up and ceil give (3, 2), truncation gives
    (2, 1). Pinning it matters because the rule sets every number in the quota table, and
    ``p_end`` is meant to be quotable across datasets.
    """
    assert 0.025 * POOL_SIZES["A"] == 2.5, "fixture no longer lands on a tie"
    assert 0.025 * POOL_SIZES["B"] == 1.5, "fixture no longer lands on a tie"
    qs, _, _ = _quotas(pool_object, ["A", "B"], ["C"], p_end=0.025, seed=42,
                       sampling_mode="random", p_basis="cluster")
    assert qs == {"A": 2, "B": 2}, "rounding is no longer half-to-even"
    # A tie above an ODD integer still rounds up, so this is half-to-even, not truncation.
    assert 0.0875 * POOL_SIZES["C"] == 3.5
    _, qk, _ = _quotas(pool_object, ["A"], ["C"], p_end=0.0875, seed=42,
                       sampling_mode="random", p_basis="cluster")
    assert qk == {"C": 4}


# --------------------------------------------------------------------------------------
# validation and forwarding
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["LCC", "clusters", "cell", "", "lcc ", None, 0.05])
def test_unknown_p_basis_raises_valueerror_naming_both(toy_graph, bad):
    """A misspelling is a caller error, and the message must name both accepted values."""
    with pytest.raises(ValueError) as e:
        dt.tl.endpoints(toy_graph, ["source"], ["sink"], cluster_key="clusters",
                        graph="intr", verbose=0, p_basis=bad)
    msg = str(e.value)
    assert "p_basis" in msg
    assert "'lcc'" in msg and "'cluster'" in msg, msg
    assert repr(bad) in msg, msg


def test_bad_p_basis_is_rejected_before_any_work(toy_graph, monkeypatch):
    """The check runs up front: a typo must not cost the caller a graph build."""
    from deltatrace import _endpoints

    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("p_basis validated too late: the graph was already built")

    monkeypatch.setattr(_endpoints, "_build_cell_graph", boom)
    with pytest.raises(ValueError, match="p_basis"):
        dt.tl.endpoints(toy_graph, ["source"], ["sink"], cluster_key="clusters",
                        graph="intr", verbose=0, p_basis="nope")
    assert not calls

    # Without this the test passes vacuously: the patched builder must be the one
    # endpoints actually calls, so a valid p_basis has to run straight into it.
    with pytest.raises(AssertionError, match="graph was already built"):
        dt.tl.endpoints(toy_graph, ["source"], ["sink"], cluster_key="clusters",
                        graph="intr", verbose=0, p_basis="lcc")
    assert len(calls) == 1


def test_trajectory_forwards_p_basis(toy_graph):
    """trajectory must hand p_basis to endpoints, not swallow it."""
    obj_c = toy_graph.copy()
    obj_l = toy_graph.copy()
    common = dict(cluster_key="clusters", graph="intr", basis="spatial",
                  p_end=0.10, seed=42, verbose=0)
    res_c = dt.tl.trajectory(obj_c, ["source"], ["mid", "sink"], p_basis="cluster",
                             **common)
    res_l = dt.tl.trajectory(obj_l, ["source"], ["mid", "sink"], p_basis="lcc", **common)
    # "cluster" at 0.10: 5 sources, 5+5 sinks. "lcc": 15 per role, split 7/8.
    assert len(res_c["endpoints"]["source_indices"]) == 5
    assert len(res_c["endpoints"]["sink_indices"]) == 10
    assert len(res_l["endpoints"]["source_indices"]) == 15
    assert len(res_l["endpoints"]["sink_indices"]) == 15
    # A forwarded value has to reach the pseudotime, or the forwarding is cosmetic.
    assert not np.allclose(obj_c.obs["pseudotime"].values, obj_l.obs["pseudotime"].values)


def test_trajectory_defaults_to_cluster(toy_graph):
    """trajectory's default must match endpoints' default, not diverge from it."""
    obj_a, obj_b = toy_graph.copy(), toy_graph.copy()
    common = dict(cluster_key="clusters", graph="intr", basis="spatial",
                  p_end=0.10, seed=42, verbose=0)
    a = dt.tl.trajectory(obj_a, ["source"], ["mid", "sink"], **common)
    b = dt.tl.trajectory(obj_b, ["source"], ["mid", "sink"], p_basis="cluster", **common)
    assert list(a["endpoints"]["source_indices"]) == list(b["endpoints"]["source_indices"])
    assert list(a["endpoints"]["sink_indices"]) == list(b["endpoints"]["sink_indices"])


def _isolated(obj, cluster):
    """A copy with every edge of `cluster` removed: its cells exist but lie off the LCC."""
    o = obj.copy()
    idx = np.flatnonzero(o.obs["clusters"].to_numpy() == cluster)
    for k in ("neighbors_intr_distances", "neighbors_intr_connectivities"):
        m = sp.lil_matrix(o.obsp[k])
        m[idx, :] = 0
        m[:, idx] = 0
        m = m.tocsr()
        m.eliminate_zeros()
        o.obsp[k] = m
    return o


def test_cluster_basis_empty_pool_is_an_error(pool_object):
    """A role cluster with nothing in the LCC cannot be sized, and must not be skipped."""
    obj = _isolated(pool_object, "rare")
    with pytest.raises(ValueError, match=r"none of its 3 cells are in the largest connected"):
        dt.tl.endpoints(obj, ["A"], ["rare"], cluster_key="clusters",
                        graph="intr", p_end=0.10, seed=42, verbose=0,
                        p_basis="cluster")


def test_an_unknown_cluster_label_is_named(pool_object):
    """A label that is not in obs[cluster_key] at all is a different error from an empty
    pool, and says which labels exist."""
    obj = pool_object.copy()
    with pytest.raises(ValueError, match=r"\['nosuchcluster'\] not found.*'rare'"):
        dt.tl.endpoints(obj, ["A"], ["nosuchcluster"], cluster_key="clusters",
                        graph="intr", p_end=0.10, seed=42, verbose=0)


# --------------------------------------------------------------------------------------
# the lcc clamp: a cluster too small for its share of the LCC
# --------------------------------------------------------------------------------------
def test_lcc_basis_clamps_a_deficient_cluster_to_its_pool(pool_object):
    """A cluster holding less of the LCC than p_end asks for cannot fill its quota however
    the cells are drawn, so it takes everything it has and says what that costs."""
    with pytest.warns(UserWarning) as rec:
        gr = dt.tl.endpoints(pool_object, ["A"], ["rare"],
                             cluster_key="clusters", graph="intr", p_end=0.10, seed=42,
                             verbose=0, p_basis="lcc")
    msg = str(rec[0].message)
    assert rec[0].filename == __file__                # points at the caller's line
    assert len(gr["sink_indices"]) == POOL_SIZES["rare"]
    assert "totals no longer match" in msg            # names what it costs
    assert "p_basis='cluster'" in msg                 # names the way out
    assert abs(gr["f"].sum()) < 1e-9                  # demand still balances


def test_lcc_basis_does_not_warn_when_every_quota_fits(pool_object):
    """The clamp has to be silent on the ordinary path, or it is just noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dt.tl.endpoints(pool_object, ["A"], ["C"], cluster_key="clusters",
                        graph="intr", p_end=0.10, seed=42, verbose=0, p_basis="lcc")


def test_lcc_basis_clamped_quota_is_monotone_in_p_end(pool_object):
    """Asking for more must never hand back fewer cells: a knob that runs backwards is
    worse than the error it replaces."""
    counts = []
    for p_end in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gr = dt.tl.endpoints(pool_object, ["A"], ["rare"],
                                 cluster_key="clusters", graph="intr", p_end=p_end,
                                 seed=42, verbose=0, p_basis="lcc")
        counts.append(len(gr["sink_indices"]))
    assert counts == sorted(counts)
    assert counts[-1] == POOL_SIZES["rare"]           # saturates at the pool, never past


def test_lcc_basis_still_refuses_an_empty_pool(pool_object):
    """Clamping to zero is not sampling: nothing in the LCC still has to be an error."""
    obj = _isolated(pool_object, "rare")
    with pytest.raises(ValueError, match=r"only 0 of its 3 are in the largest connected"):
        dt.tl.endpoints(obj, ["A"], ["rare"], cluster_key="clusters",
                        graph="intr", p_end=0.10, seed=42, verbose=0, p_basis="lcc")


def test_both_entry_points_default_to_the_cluster_basis():
    """endpoints and trajectory both default to "cluster", the benchmarked basis: it gives
    every declared cluster at least one cell, where "lcc" can leave a cluster with none."""
    import inspect

    defaults = {f.__name__: inspect.signature(f).parameters["p_basis"].default
                for f in (dt.tl.endpoints, dt.tl.trajectory)}
    assert defaults == {"endpoints": "cluster", "trajectory": "cluster"}
