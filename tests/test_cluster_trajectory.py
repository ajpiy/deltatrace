"""tl/pl.cluster_trajectory — the PAGA-free cluster graph read off the QROT flow.

The flows here are hand-built rather than solved, so every number in the assertions can be
worked out on paper: the net-flux rule, the theta cutoff and the acyclicity guard are each
pinned against an independent recomputation instead of against a stored snapshot.

The guard is the one that matters. The cell-level flow is a DAG in every fixture below,
yet ``_reentrant`` still produces a directed CYCLE at cluster level, because a cluster is
not a point in the potential the cells are ordered by.
"""

import itertools

import anndata as ad
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")                        # no display in CI; must precede pyplot
import matplotlib.pyplot as plt

import deltatrace as dt
from deltatrace import _lineages


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _obj(clusters, pt, edges, J, coords=None):
    """An AnnData carrying nothing but cluster labels, pseudotime and a hand-built flow."""
    n = len(clusters)
    obj = ad.AnnData(
        X=np.zeros((n, 1)),
        obs=pd.DataFrame({"clusters": list(clusters), "pseudotime": np.asarray(pt, float)},
                         index=[f"c{i}" for i in range(n)]),
    )
    obj.obsm["spatial"] = (np.asarray(coords, float) if coords is not None
                           else np.column_stack([np.asarray(pt, float), np.zeros(n)]))
    obj.uns["dt_flow"] = {"J": np.asarray(J, float),
                          "edges": np.asarray(edges, np.int64)}
    return obj


@pytest.fixture
def forked():
    """A source that splits into two branches of unequal mass, plus a thin cross-link.

        A --3.0--> B --2.9--> D
        A --1.0--> C --1.1--> E
        B --0.1--> C            (the thin one theta is meant to cut)

    Mass conserves at every cluster, so M is exactly the 4.0 that leaves A.
    """
    return _obj(
        clusters=["A", "B", "C", "D", "E"],
        pt=[0.0, 0.4, 0.4, 1.0, 1.0],
        edges=[(0, 1), (1, 3), (0, 2), (2, 4), (1, 2)],
        J=[3.0, 2.9, 1.0, 1.1, 0.1],
    )


@pytest.fixture
def reentrant():
    """A cluster that is NON-CONVEX in pseudotime: the flow leaves A and comes back to it.

    Cells (pseudotime in brackets) form one strictly increasing chain, so the CELL-level
    flow is a DAG::

        s[0.0] -> a1[0.2] -> b1[0.4] -> c1[0.6] -> a2[0.8] -> t[1.0]
         (S)        (A)        (B)        (C)        (A)       (T)

    but a1 and a2 are both cluster A, so at CLUSTER level the net flux contains the cycle
    A -> B -> C -> A. Nothing about the cell-level DAG prevents it.
    """
    return _obj(
        clusters=["S", "A", "B", "C", "A", "T"],
        pt=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        J=[1.0, 1.0, 1.0, 1.0, 1.0],
    )


@pytest.fixture
def at_cut():
    """Two edges out of one source, chosen so a weight lands EXACTLY on the theta cut.

        A --3.0--> B
        A --1.0--> C

    M = 0.5 * (4 + 3 + 1) = 4.0 exactly, so theta = 0.25 puts the cut at 1.0, which is
    A -> C's weight to the bit. The rule is ``>=``, so that edge must survive.
    """
    return _obj(
        clusters=["A", "A", "B", "C"],
        pt=[0.0, 0.0, 1.0, 1.0],
        edges=[(0, 2), (1, 3)],
        J=[3.0, 1.0],
    )


@pytest.fixture
def skewed_pt():
    """A cluster whose pseudotime median and mean fall on opposite sides of its successor.

    A holds 0.10, 0.20, 0.90: median 0.20, mean 0.40. B sits at 0.30. So A -> B runs
    FORWARD in median pseudotime and BACKWARD in mean pseudotime, which is what makes the
    median-vs-mean choice observable in ``n_median_inverted``.
    """
    return _obj(
        clusters=["A", "A", "A", "B", "B"],
        pt=[0.10, 0.20, 0.90, 0.30, 0.30],
        edges=[(2, 3)],
        J=[1.0],
    )


@pytest.fixture
def convex():
    """The ``reentrant`` chain with the re-entrant cell relabeled: no cycle to break."""
    return _obj(
        clusters=["S", "A", "B", "C", "D", "T"],
        pt=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        J=[1.0, 1.0, 1.0, 1.0, 1.0],
    )


def _net_by_hand(obj, names, edge_threshold=1e-6):
    """Recompute F, NET and M straight from the flow, independently of the package."""
    lab = obj.obs["clusters"].astype(str).to_numpy()
    idx = {c: i for i, c in enumerate(names)}
    E = np.asarray(obj.uns["dt_flow"]["edges"], np.int64)
    J = np.asarray(obj.uns["dt_flow"]["J"], float)
    F = np.zeros((len(names), len(names)))
    for (u, v), j in zip(E, J, strict=True):
        if j > edge_threshold and lab[u] != lab[v]:
            F[idx[lab[u]], idx[lab[v]]] += j
    NET = F - F.T
    return NET, 0.5 * float(np.abs(NET.sum(axis=1)).sum())


def _pairs(res):
    return {(r.src, r.dst) for r in res["edges"].itertuples(index=False)}


def _pairs_ordered(res):
    return [(r.src, r.dst) for r in res["edges"].itertuples(index=False)]


# --------------------------------------------------------------- the net-flux rule
def test_edges_match_the_net_flux_rule(forked):
    """Retained edges are exactly {i -> j : NET[i,j] >= theta * M}, recomputed by hand."""
    res = dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=0.05, verbose=0)
    names = res["names"]
    assert names == ["A", "B", "C", "D", "E"]

    NET, M = _net_by_hand(forked, names)
    assert np.allclose(res["net_flux"], NET)
    assert res["M"] == pytest.approx(M)
    assert M == pytest.approx(4.0)           # 4.0 leaves A, 2.9 + 1.1 arrives in D + E

    expected = {(names[i], names[j])
                for i in range(len(names)) for j in range(len(names))
                if NET[i, j] >= 0.05 * M}
    assert _pairs(res) == expected
    assert expected == {("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")}

    # weight IS the net flux the edge carries, and frac is it as a share of M
    loose = dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=0.01, verbose=0)
    w = {(r.src, r.dst): r.weight for r in loose["edges"].itertuples(index=False)}
    fr = {(r.src, r.dst): r.frac for r in loose["edges"].itertuples(index=False)}
    assert w[("A", "B")] == pytest.approx(3.0)
    assert w[("B", "C")] == pytest.approx(0.1)
    assert fr[("B", "C")] == pytest.approx(0.1 / 4.0)


def test_net_flux_is_antisymmetric_and_orientation_comes_from_its_sign(forked):
    res = dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    NET = res["net_flux"]
    assert np.allclose(NET, -NET.T)
    for r in res["edges"].itertuples(index=False):
        if not r.flipped:
            assert NET[r.i, r.j] > 0.0


def test_throughput_is_max_of_net_in_and_out(forked):
    """Node sizing must be the mass through the cluster, not its cell count."""
    res = dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    pos = np.maximum(res["net_flux"], 0.0)
    assert np.allclose(res["out_flux"], pos.sum(axis=1))
    assert np.allclose(res["in_flux"], pos.sum(axis=0))
    assert np.allclose(res["throughput"], np.maximum(pos.sum(axis=0), pos.sum(axis=1)))
    tp = dict(zip(res["names"], res["throughput"], strict=True))
    assert tp["A"] == pytest.approx(4.0)     # the source passes all of M on
    assert tp["D"] == pytest.approx(2.9)


# --------------------------------------------------------------- theta thresholding
def test_raising_theta_drops_edges_monotonically(forked):
    """More theta, never more edges, and the surviving set is always a subset."""
    sets = []
    for theta in (0.01, 0.05, 0.2, 0.5, 0.7):
        res = dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=theta,
                                       repair=False, verbose=0)
        sets.append(_pairs(res))
    for lo, hi in itertools.pairwise(sets):
        assert hi <= lo, "raising theta must only ever remove edges"
    assert [len(s) for s in sets] == [5, 4, 4, 2, 2]

    # the thin B -> C link (0.1 of M = 4.0, i.e. 2.5%) is the first to go
    assert ("B", "C") in sets[0]
    assert ("B", "C") not in sets[1]


# --------------------------------------------------------------- connectivity repair
def test_repair_keeps_graph_connected(forked):
    """repair re-admits sub-theta edges only to merge components, never within one."""
    import networkx as nx
    for theta in (0.01, 0.05, 0.2, 0.5, 0.7):
        bare = dt.tl.cluster_trajectory(forked, "clusters", theta=theta,
                                        repair=False, verbose=0)
        fixed = dt.tl.cluster_trajectory(forked, "clusters", theta=theta,
                                         repair=True, verbose=0)
        assert _pairs(bare) <= _pairs(fixed)
        assert fixed["n_repair_edges_added"] == len(_pairs(fixed)) - len(_pairs(bare))
        G = nx.Graph()
        G.add_nodes_from(range(len(fixed["names"])))
        G.add_edges_from(zip(fixed["edges"]["i"], fixed["edges"]["j"], strict=True))
        assert nx.number_connected_components(G) == 1, f"disconnected at theta={theta}"


def test_an_edge_exactly_on_the_theta_cut_is_kept(at_cut):
    """``NET[i,j] >= theta * M``, not ``>``: the documented rule, pinned at the boundary.

    Monotonicity alone cannot see this, since a nested chain of subsets stays nested under
    either comparison. Only a weight sitting ON the cut separates them.
    """
    res = dt.tl.cluster_trajectory(at_cut, "clusters", edge_rule="theta", theta=0.25,
                                   repair=False, verbose=0)
    assert res["M"] == pytest.approx(4.0)
    assert _pairs(res) == {("A", "B"), ("A", "C")}
    weights = dict(zip(_pairs_ordered(res), res["edges"]["weight"], strict=True))
    assert weights[("A", "C")] == pytest.approx(0.25 * res["M"])   # exactly on the cut

    # a hair above and the same edge goes; a hair below and nothing changes
    assert _pairs(dt.tl.cluster_trajectory(at_cut, "clusters", edge_rule="theta",
                                           theta=0.2500001, repair=False,
                                           verbose=0)) == {("A", "B")}
    assert _pairs(dt.tl.cluster_trajectory(at_cut, "clusters", edge_rule="theta",
                                           theta=0.2499999, repair=False,
                                           verbose=0)) == {("A", "B"), ("A", "C")}


def test_pruned_edges_are_counted_not_silently_dropped(forked):
    """kept == candidates - pruned, so no edge ever vanishes unaccounted for."""
    for theta in (0.01, 0.05, 0.2, 0.5, 0.7):
        res = dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=theta,
                                       repair=False, verbose=0)
        assert len(res["edges"]) == res["n_candidate_pairs"] - res["n_pruned_by_theta"]
        assert res["theta"] == pytest.approx(theta)
    # with repair off, a theta that fragments the graph says so
    assert dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=0.7,
                                    repair=False, verbose=0)["n_components"] == 3


# --------------------------------------------------------------- the full pipeline
@pytest.fixture
def leaky_fork():
    """Two leaky branches converging on one target: the prune's visiting order decides.

        A --1.0--> B --0.2--> D
        A --1.0--> C --0.3--> D

    B and C both retain most of what reaches them (0.2 and 0.3 of 1.0 leave), so both
    B -> D and C -> D are leak candidates, but D must keep one. Visiting in cluster-index
    order drops B -> D first and then has to keep C -> D; heaviest-first would do the
    reverse. The package visits in index order.
    """
    return _obj(
        clusters=["A", "B", "C", "D"],
        pt=[0.0, 0.5, 0.5, 1.0],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        J=[1.0, 1.0, 0.2, 0.3],
    )


def test_leak_prune_visits_candidates_in_cluster_index_order(leaky_fork):
    res = dt.tl.cluster_trajectory(leaky_fork, "clusters", verbose=0)
    assert res["n_leak_pruned"] == 1
    assert _pairs(res) == {("A", "B"), ("A", "C"), ("C", "D")}   # not B -> D


def test_default_pipeline_is_prune_then_meg_then_repair_then_guard(forked):
    """MEG drops A -> C because A -> B -> C already implies it, whatever its mass;
    repair is on by default but has nothing to re-admit after a reachability-
    preserving reduction."""
    res = dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    assert res["simplify"] is True
    assert _pairs(res) == {("A", "B"), ("B", "C"), ("B", "D"), ("C", "E")}
    assert res["n_repair_edges_added"] == 0
    assert res["n_components"] == 1


def test_simplify_false_stops_after_the_leak_prune(forked, leaky_fork):
    """Prune-only mode keeps every surviving net-flux candidate, MEG or not."""
    res = dt.tl.cluster_trajectory(forked, "clusters", simplify=False, verbose=0)
    assert res["simplify"] is False
    assert res["n_leak_pruned"] == 0
    assert _pairs(res) == {("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E")}

    res = dt.tl.cluster_trajectory(leaky_fork, "clusters", simplify=False, verbose=0)
    assert res["n_leak_pruned"] == 1                    # the prune still runs
    assert _pairs(res) == {("A", "B"), ("A", "C"), ("C", "D")}


def test_simplify_false_still_runs_the_acyclicity_guard(reentrant):
    res = dt.tl.cluster_trajectory(reentrant, "clusters", simplify=False, verbose=0)
    g = nx.DiGraph()
    g.add_edges_from((r.i, r.j) for r in res["edges"].itertuples(index=False))
    assert nx.is_directed_acyclic_graph(g)
    assert res["n_cycle_flips"] == 1


def test_repair_is_on_by_default(forked):
    res = dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=0.7, verbose=0)
    assert res["n_repair_edges_added"] == 2             # C and E re-attached
    assert res["n_components"] == 1


# --------------------------------------------------------------- the acyclicity guard
def test_the_raw_cluster_graph_really_is_cyclic(reentrant):
    """The premise: an acyclic CELL flow whose CLUSTER net flux contains a cycle."""
    E = np.asarray(reentrant.uns["dt_flow"]["edges"], np.int64)
    cell = nx.DiGraph(); cell.add_edges_from(map(tuple, E))
    assert nx.is_directed_acyclic_graph(cell), "the cell-level flow must be a DAG"

    res = dt.tl.cluster_trajectory(reentrant, "clusters", verbose=0)
    names = res["names"]
    NET, _M = _net_by_hand(reentrant, names)
    raw = nx.DiGraph()
    raw.add_nodes_from(range(len(names)))
    raw.add_edges_from((i, j) for i in range(len(names)) for j in range(len(names))
                       if NET[i, j] > 0)
    assert not nx.is_directed_acyclic_graph(raw)
    assert set(map(tuple, nx.find_cycle(raw))) == {(0, 1), (1, 2), (2, 0)}   # A -> B -> C -> A


def test_acyclicity_guard_fires_and_reports_it(reentrant):
    res = dt.tl.cluster_trajectory(reentrant, "clusters", verbose=0)

    g = nx.DiGraph()
    g.add_nodes_from(range(len(res["names"])))
    g.add_edges_from((r.i, r.j) for r in res["edges"].itertuples(index=False))
    assert nx.is_directed_acyclic_graph(g), "cluster_trajectory must never return a cycle"

    assert res["n_cycle_flips"] == 1
    assert res["n_cycle_unresolved"] == 0
    assert res["edges"]["flipped"].sum() == 1
    # C -> A closed the cycle and was reversed; the pair survives, its direction does not
    flipped = res["edges"].loc[res["edges"]["flipped"], ["src", "dst"]]
    assert list(flipped.itertuples(index=False, name=None)) == [("A", "C")]


def test_median_inversion_alarm_counts_the_backwards_edges(reentrant):
    """The independent detector: kept edges running backwards in median pseudotime."""
    res = dt.tl.cluster_trajectory(reentrant, "clusters", verbose=0)
    pt = dict(zip(res["names"], res["cluster_pt"], strict=True))
    assert pt["A"] == pytest.approx(0.5)     # median(0.2, 0.8): NOT between B and C
    assert res["n_median_inverted"] == 2     # A -> B and C -> A both go back in time


def test_median_inversion_is_counted_before_the_guard_not_after(reentrant):
    """The alarm reads the RAW NET orientation, which is the documented definition.

    Counting it after the guard would make it a readout of the guard instead of a check on
    it: every flip the guard performs removes one inversion by construction, so a
    post-guard count could only ever agree with itself.
    """
    res = dt.tl.cluster_trajectory(reentrant, "clusters", verbose=0)
    pt = dict(zip(res["names"], res["cluster_pt"], strict=True))

    after = sum(1 for s, d in _pairs_ordered(res) if pt[s] > pt[d])
    assert after == 1                        # only A -> B survives backwards
    assert res["n_median_inverted"] == 2     # A -> B and the pre-flip C -> A
    assert res["n_median_inverted"] - after == res["n_cycle_flips"]


def test_tied_medians_hide_an_inversion_from_the_alarm_but_not_from_the_guard():
    """``n_median_inverted == 0`` only implies "already a DAG" when the medians differ.

    Flatten the re-entrant chain's interior to one tied pseudotime and the cycle is still
    there, the guard still breaks it, and the alarm still reads 0. Which is why the two
    counters have to be read together.
    """
    obj = _obj(
        clusters=["S", "A", "B", "C", "A", "T"],
        pt=[0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        J=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    res = dt.tl.cluster_trajectory(obj, "clusters", verbose=0)
    pt = dict(zip(res["names"], res["cluster_pt"], strict=True))
    assert pt["A"] == pt["B"] == pt["C"] == pytest.approx(0.5)
    assert res["n_cycle_flips"] == 1          # the guard had to act
    assert res["n_median_inverted"] == 0      # and the alarm could not see it


def test_cluster_pseudotime_is_the_median_not_the_mean(skewed_pt):
    """The per-cluster summary feeding node color and the alarm is a median.

    A = {0.10, 0.20, 0.90} against B = 0.30: median 0.20 puts A -> B forward, mean 0.40
    would put it backward and raise a false alarm.
    """
    res = dt.tl.cluster_trajectory(skewed_pt, "clusters", verbose=0)
    pt = dict(zip(res["names"], res["cluster_pt"], strict=True))
    assert pt["A"] == pytest.approx(0.20)     # mean would be 0.40
    assert pt["B"] == pytest.approx(0.30)
    assert _pairs(res) == {("A", "B")}
    assert res["n_median_inverted"] == 0      # 1 under a mean


def test_guard_is_a_no_op_on_a_convex_clustering(convex):
    """Control: relabel the re-entrant cell and both counters go to zero."""
    res = dt.tl.cluster_trajectory(convex, "clusters", verbose=0)
    assert res["n_cycle_flips"] == 0
    assert res["n_cycle_unresolved"] == 0
    assert res["n_median_inverted"] == 0
    assert not res["edges"]["flipped"].any()
    assert _pairs(res) == {("S", "A"), ("A", "B"), ("B", "C"), ("C", "D"), ("D", "T")}


def test_enforce_acyclic_false_lets_a_real_cycle_survive(reentrant):
    """The re-entrant fixture's cluster-level cycle is genuine, not a fixture quirk.

    With the guard off, A -> B -> C -> A must all come back exactly as NET orients
    them: no flips, no unresolved (nothing was ever attempted), and the resulting
    graph is NOT a DAG -- the opposite of test_acyclicity_guard_fires_and_reports_it.
    """
    res = dt.tl.cluster_trajectory(reentrant, "clusters", enforce_acyclic=False, verbose=0)

    assert res["enforce_acyclic"] is False
    assert res["n_cycle_flips"] == 0
    assert res["n_cycle_unresolved"] == 0
    assert not res["edges"]["flipped"].any()
    assert _pairs(res) == {("S", "A"), ("A", "B"), ("B", "C"), ("C", "A"), ("A", "T")}

    g = nx.DiGraph()
    g.add_nodes_from(range(len(res["names"])))
    g.add_edges_from((r.i, r.j) for r in res["edges"].itertuples(index=False))
    assert not nx.is_directed_acyclic_graph(g), "the whole point: a real cycle survives"


def test_enforce_acyclic_true_is_still_the_default(reentrant):
    """No behavior change for existing callers who never pass the new argument."""
    res = dt.tl.cluster_trajectory(reentrant, "clusters", verbose=0)
    assert res["enforce_acyclic"] is True


def test_median_inversion_reports_minus_one_without_pseudotime(reentrant):
    obj = reentrant.copy()
    del obj.obs["pseudotime"]
    res = dt.tl.cluster_trajectory(obj, "clusters", verbose=0)
    assert res["n_median_inverted"] == -1
    assert np.isnan(res["cluster_pt"]).all()
    assert res["n_cycle_flips"] == 1         # the guard itself never needed pseudotime


def test_median_inversion_reports_minus_one_when_one_cluster_has_no_finite_pseudotime():
    """-1 means NOT RUN, and a present-but-partly-NaN column cannot run it either."""
    obj = _obj(
        clusters=["A", "A", "B", "B", "C", "C"],
        pt=[0.0, 0.2, np.nan, np.nan, 0.8, 1.0],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        J=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    res = dt.tl.cluster_trajectory(obj, "clusters", verbose=0)
    pt = dict(zip(res["names"], res["cluster_pt"], strict=True))
    assert np.isnan(pt["B"])                 # the column is there, this cluster is not
    assert res["n_median_inverted"] == -1    # NOT RUN, which is not the same as 0


# --------------------------------------------------------------- input validation
def test_raises_when_fewer_than_two_clusters_carry_flux():
    obj = _obj(clusters=["A", "A", "A"], pt=[0.0, 0.5, 1.0],
               edges=[(0, 1), (1, 2)], J=[1.0, 1.0])
    with pytest.raises(ValueError, match="carry cross-cluster flux"):
        dt.tl.cluster_trajectory(obj, "clusters", verbose=0)


def test_raises_when_the_edge_threshold_silences_every_boundary(forked):
    with pytest.raises(ValueError, match="carry cross-cluster flux"):
        dt.tl.cluster_trajectory(forked, "clusters", edge_threshold=1e3, verbose=0)


@pytest.mark.parametrize("theta", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_raises_on_theta_outside_the_open_unit_interval(forked, theta):
    """theta validation only fires under edge_rule="theta": "meg" never reads theta."""
    with pytest.raises(ValueError, match="strictly inside"):
        dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=theta, verbose=0)


def test_raises_on_a_non_numeric_theta(forked):
    with pytest.raises(ValueError, match="must be a number"):
        dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta="0.05", verbose=0)


def test_raises_when_the_flow_is_missing(forked):
    obj = forked.copy()
    del obj.uns["dt_flow"]
    with pytest.raises(ValueError, match=r"No flow data in adata.uns\['dt_flow'\]"):
        dt.tl.cluster_trajectory(obj, "clusters", verbose=0)


def test_raises_on_a_key_added_whose_flow_was_never_written(forked):
    with pytest.raises(ValueError, match=r"uns\['dt_flow_run2'\]"):
        dt.tl.cluster_trajectory(forked, "clusters", key_added="run2", verbose=0)


def test_no_bare_assert_guards_the_new_code_paths():
    """python -O strips assert, so a rejection written as one silently stops rejecting."""
    import ast
    import inspect


    for fn in (dt.tl.cluster_trajectory, dt.pl.cluster_trajectory, _lineages.cluster_graph,
               _lineages.net_flux_matrix, _lineages.cluster_flux_matrix,
               _lineages._cross_cluster_flux, _lineages._median_cluster_pt):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        offenders = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
        assert not offenders, f"{fn.__name__} guards with assert at {offenders}"


# --------------------------------------------------------------- key_added round-trip
def test_result_is_written_to_uns_and_returned_unchanged(forked):
    res = dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    assert forked.uns["dt_cluster_trajectory"] is res


def test_key_added_round_trips_and_two_runs_coexist(forked):
    obj = forked.copy()
    obj.uns["dt_flow_run2"] = obj.uns["dt_flow"]
    obj.obs["pseudotime_run2"] = obj.obs["pseudotime"].to_numpy(float)

    loose = dt.tl.cluster_trajectory(obj, "clusters", edge_rule="theta", theta=0.01, verbose=0)
    tight = dt.tl.cluster_trajectory(obj, "clusters", edge_rule="theta", theta=0.5,
                                     key_added="run2", verbose=0)

    assert "dt_cluster_trajectory" in obj.uns
    assert "dt_cluster_trajectory_run2" in obj.uns
    assert obj.uns["dt_cluster_trajectory"] is loose
    assert obj.uns["dt_cluster_trajectory_run2"] is tight
    assert _pairs(loose) != _pairs(tight)    # the second run did not clobber the first
    assert obj.uns["dt_cluster_trajectory"]["theta"] == pytest.approx(0.01)


def test_rerunning_into_the_same_key_overwrites_it(forked):
    """A second run must REPLACE the stored graph, not leave a stale one behind it.

    Serving the first result back would hand the user a graph computed at a theta they no
    longer asked for, and every plot off it would be silently wrong.
    """
    first = dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=0.05,
                                     repair=False, verbose=0)
    assert forked.uns["dt_cluster_trajectory"] is first

    second = dt.tl.cluster_trajectory(forked, "clusters", edge_rule="theta", theta=0.40,
                                      repair=False, verbose=0)
    stored = forked.uns["dt_cluster_trajectory"]
    assert stored is second
    assert stored is not first
    assert stored["theta"] == pytest.approx(0.40)
    assert _pairs(stored) == _pairs(second) != _pairs(first)


def test_result_records_the_inputs_it_was_computed_from(forked):
    """cluster_key / pt_key / edge_threshold travel with the graph, for pl to check."""
    res = dt.tl.cluster_trajectory(forked, "clusters", edge_threshold=1e-9, verbose=0)
    assert res["cluster_key"] == "clusters"
    assert res["pt_key"] == "pseudotime"
    assert res["edge_threshold"] == pytest.approx(1e-9)

    obj = forked.copy()
    obj.uns["dt_flow_run2"] = obj.uns["dt_flow"]
    tagged = dt.tl.cluster_trajectory(obj, "clusters", key_added="run2", verbose=0)
    assert tagged["pt_key"] == "pseudotime_run2"


def test_key_added_only_writes_the_suffixed_key(forked):
    obj = forked.copy()
    obj.uns["dt_flow_run2"] = obj.uns["dt_flow"]
    dt.tl.cluster_trajectory(obj, "clusters", key_added="run2", verbose=0)
    assert "dt_cluster_trajectory" not in obj.uns


# --------------------------------------------------------------- pl.cluster_trajectory
def test_plot_returns_a_figure(forked):
    dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    fig, ax = dt.pl.cluster_trajectory(forked, basis="spatial", show=False)
    assert fig is ax.figure
    assert len(ax.collections) == 1          # the node scatter


def test_plot_honors_a_caller_supplied_ax(forked):
    dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    fig, (left, right) = plt.subplots(1, 2)
    out_fig, out_ax = dt.pl.cluster_trajectory(forked, basis="spatial", ax=right,
                                               show=False)
    assert out_ax is right
    assert out_fig is fig
    assert not left.collections               # nothing was drawn on the other panel


def test_plot_puts_the_nodes_at_the_cluster_median_not_the_mean():
    """One outlying off-LCC cell must not drag its cluster's node across the embedding."""
    obj = _obj(
        clusters=["A", "A", "A", "B"],
        pt=[0.0, 0.0, 0.0, 1.0],
        edges=[(0, 3)],
        J=[1.0],
        coords=[[0.0, 0.0], [1.0, 0.0], [1000.0, 0.0], [2.0, 0.0]],   # cell 2 is far off
    )
    dt.tl.cluster_trajectory(obj, "clusters", verbose=0)
    _fig, ax = dt.pl.cluster_trajectory(obj, basis="spatial", show=False)
    pos = ax.collections[0].get_offsets()
    assert pos[0][0] == pytest.approx(1.0)   # median(0, 1, 1000); the mean would be ~333
    assert pos[1][0] == pytest.approx(2.0)


def test_plot_saves_the_supplied_figure(forked, tmp_path, monkeypatch):
    """save must hit the figure we drew on, not the pyplot current one."""
    from matplotlib.figure import Figure

    seen = []
    real = Figure.savefig
    monkeypatch.setattr(Figure, "savefig",
                        lambda self, *a, **k: (seen.append(self), real(self, *a, **k))[1])

    dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    target_fig, target_ax = plt.subplots()
    plt.figure()                              # a DIFFERENT current figure
    out = tmp_path / "ct.png"
    dt.pl.cluster_trajectory(forked, basis="spatial", ax=target_ax, save=str(out),
                             show=False)
    assert out.exists()
    assert seen == [target_fig]


def test_plot_raises_when_the_graph_was_never_computed(forked):
    with pytest.raises(ValueError, match=r"run deltatrace\.tl\.cluster_trajectory"):
        dt.pl.cluster_trajectory(forked, basis="spatial", show=False)


def test_plot_raises_on_a_cluster_key_the_graph_was_not_computed_on(forked):
    dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    forked.obs["other"] = ["X"] * forked.n_obs
    with pytest.raises(ValueError, match="was computed on"):
        dt.pl.cluster_trajectory(forked, basis="spatial", cluster_key="other", show=False)


def test_plot_rejects_a_decoy_column_carrying_the_same_labels(forked):
    """The dangerous case: a DIFFERENT column reusing the same label set.

    Checking only that the labels are present passes here, and every node then lands on a
    different set of cells: a plausible figure, entirely wrong. The graph records the
    column it was computed on so this is refused instead.
    """
    dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    rot = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"}
    forked.obs["permuted"] = [rot[c] for c in forked.obs["clusters"]]
    assert set(forked.obs["permuted"]) == set(forked.obs["clusters"])   # same labels

    with pytest.raises(ValueError, match=r"cluster_key='permuted'.*computed on 'clusters'"):
        dt.pl.cluster_trajectory(forked, basis="spatial", cluster_key="permuted",
                                 show=False)


def test_plot_accepts_the_column_the_graph_records(forked):
    """Control for the check above: the matching column still draws."""
    forked.obs["mine"] = forked.obs["clusters"].to_numpy()
    dt.tl.cluster_trajectory(forked, "mine", verbose=0)
    _fig, ax = dt.pl.cluster_trajectory(forked, basis="spatial", cluster_key="mine",
                                        show=False)
    assert len(ax.collections) == 1


def test_plot_raises_when_no_basis_can_be_resolved(forked):
    dt.tl.cluster_trajectory(forked, "clusters", verbose=0)
    with pytest.raises(ValueError, match="basis is None"):
        dt.pl.cluster_trajectory(forked, show=False)


def test_plot_dashes_the_edges_the_guard_reversed(reentrant):
    res = dt.tl.cluster_trajectory(reentrant, "clusters", verbose=0)
    assert res["n_cycle_flips"] == 1
    _fig, ax = dt.pl.cluster_trajectory(reentrant, basis="spatial", show=False)
    styles = [ln.get_linestyle() for ln in ax.lines]
    assert styles.count("--") == 1
    assert len(styles) == len(res["edges"])


# --------------------------------------------------------------- shared refactor
def test_terminality_and_the_cluster_graph_read_one_edge_frame(forked):
    """The in/out totals behind Term_i are the row/column sums of the same gross flux."""

    tab = _lineages.cluster_terminality(forked, None, "clusters")
    F, index = _lineages.cluster_flux_matrix(forked, "clusters",
                                          _lineages._flow_cluster_names(forked, "clusters"))
    for name, row in tab.iterrows():
        i = index[str(name)]
        assert row["inf"] == pytest.approx(F[:, i].sum())
        assert row["out"] == pytest.approx(F[i, :].sum())
