"""Source-only terminal discovery cut by COUNT (``n_lineages``) instead of by value.

The two knobs read the same terminality ranking, so every test here pins either the
selection they agree on or the guard rails that keep them from being combined. Flows are
hand-written ``uns['dt_flow']`` dicts so the terminality scores (and the exact ties) are known
without solving anything.
"""

import re
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from _line_toy import line_toy

import deltatrace as dt
from deltatrace import _lineages


def _ranked_flow():
    """src -> mid -> {lo, hi}, with lo passing a quarter of its mass on to hi.

    Cross-cluster flux gives an unambiguous ranking, no ties:
        hi   in 0.75, out 0     -> term_score 1.0
        lo   in 0.5,  out 0.25  -> term_score 0.5
        mid  in 1.0,  out 1.0   -> term_score 0.0
        src  in 0               -> dropped (terminality undefined)
    """
    obs = pd.DataFrame({"clusters": ["src", "mid", "lo", "hi"]},
                       index=["c0", "c1", "c2", "c3"])
    a = ad.AnnData(X=np.zeros((4, 2), np.float32), obs=obs)
    a.obs["pseudotime"] = [0.0, 0.5, 0.7, 1.0]
    a.uns["dt_flow"] = {
        "edges": np.array([[0, 1], [1, 2], [1, 3], [2, 3]], np.int64),
        "J": np.array([1.0, 0.5, 0.5, 0.25], float),
    }
    return a


def _tied_flow(w1=0.6, w2=0.4, names=("t1", "t2")):
    """src -> mid -> {t1, t2}: two sinks with NO outflow, so both score term_score == 1.0 exactly.

    ``mid`` scores 0.0, so a cut at n_lineages=1 lands inside the t1/t2 tie while
    n_lineages=2 clears it.
    """
    obs = pd.DataFrame({"clusters": ["src", "mid", *names]},
                       index=["c0", "c1", "c2", "c3"])
    a = ad.AnnData(X=np.zeros((4, 2), np.float32), obs=obs)
    a.obs["pseudotime"] = [0.0, 0.5, 1.0, 1.0]
    a.uns["dt_flow"] = {
        "edges": np.array([[0, 1], [1, 2], [1, 3]], np.int64),
        "J": np.array([w1 + w2, w1, w2], float),
    }
    return a


def test_terminality_ranking_of_the_toy_flow():
    """The terminality scores the count cut is supposed to be reading."""
    tab = dt.tl.terminality(_ranked_flow(), cluster_key="clusters",
                            source_clusters=["src"], show=False)
    assert list(tab.index) == ["hi", "lo", "mid"]
    np.testing.assert_allclose(tab["term_score"].to_numpy(), [1.0, 0.5, 0.0], atol=1e-12)


@pytest.mark.parametrize(("n", "expected"), [(1, ["hi"]), (2, ["hi", "lo"]),
                                             (3, ["hi", "lo", "mid"])])
def test_n_lineages_takes_the_top_l_clusters(n, expected):
    a = _ranked_flow()
    dt.tl.lineages(a, ["src"], None, cluster_key="clusters", n_lineages=n, verbose=False)
    assert sorted(map(str, a.uns["path_lineages"])) == sorted(expected)


def test_n_lineages_matches_the_equivalent_omega_term():
    """Both knobs cut the same ranking, so a count and a cutoff that agree must agree."""
    by_count = _ranked_flow()
    dt.tl.lineages(by_count, ["src"], None, cluster_key="clusters", n_lineages=2,
                   verbose=False)
    by_value = _ranked_flow()
    dt.tl.lineages(by_value, ["src"], None, cluster_key="clusters", omega_term=0.25,
                   verbose=False)
    assert (sorted(map(str, by_count.uns["path_lineages"]))
            == sorted(map(str, by_value.uns["path_lineages"])) == ["hi", "lo"])
    np.testing.assert_allclose(by_count.obsm["path_cellweights"],
                               by_value.obsm["path_cellweights"], atol=1e-12)


def test_n_lineages_and_omega_term_are_mutually_exclusive():
    a = _ranked_flow()
    with pytest.raises(ValueError, match="omega_term or n_lineages, not both"):
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", omega_term=0.5,
                       n_lineages=1, verbose=False)


def test_n_lineages_with_explicit_terminals_raises():
    """The terminals are already decided; silently ignoring the count would be worse."""
    a = _ranked_flow()
    with pytest.raises(ValueError, match="only applies in source-only mode"):
        dt.tl.lineages(a, ["src"], ["hi"], cluster_key="clusters", n_lineages=1,
                       verbose=False)


@pytest.mark.parametrize("bad", [0, -1, 1.5, 2.0, "2", True])
def test_n_lineages_must_be_a_positive_integer(bad):
    a = _ranked_flow()
    with pytest.raises(ValueError, match="n_lineages must be an integer >= 1"):
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", n_lineages=bad,
                       verbose=False)


def test_n_lineages_above_the_rankable_count_raises():
    """4 clusters but only 3 rankable: 'src' has no inflow and is a declared source."""
    a = _ranked_flow()
    with pytest.raises(ValueError, match=r"exceeds the 3 cluster\(s\) in the terminality ranking"):
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", n_lineages=4,
                       verbose=False)


def test_straddling_tie_warns_and_still_picks_deterministically():
    """t1 and t2 both score exactly 1.0; n_lineages=1 must take one and SAY it did."""
    a = _tied_flow(0.6, 0.4)
    with pytest.warns(UserWarning, match="tied at term_score"):
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", n_lineages=1,
                       verbose=False)
    assert list(map(str, a.uns["path_lineages"])) == ["t1"]     # more absorbed mass wins


def test_tie_broken_by_name_when_the_mass_ties_too():
    a = _tied_flow(0.5, 0.5, names=("tb", "ta"))
    with pytest.warns(UserWarning, match="tb.*ta|ta.*tb"):
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", n_lineages=1,
                       verbose=False)
    assert list(map(str, a.uns["path_lineages"])) == ["ta"]     # name ascending


def test_tie_break_prefers_the_more_absorbed_mass_against_the_incoming_order():
    """W descending must beat the order the ranking arrived in, not merely agree with it.

    ``ta`` absorbs 0.4 and ``tb`` 0.6, so W puts ``tb`` first while the alphabetical /
    construction order puts ``ta`` first: any implementation that takes the head of the
    table without re-breaking the tie picks ``ta``.
    """
    a = _tied_flow(0.4, 0.6, names=("ta", "tb"))
    with pytest.warns(UserWarning, match="tied at term_score"):
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", n_lineages=1,
                       verbose=False)
    assert list(map(str, a.uns["path_lineages"])) == ["tb"]


def test_top_terminals_reorders_whatever_frame_it_is_handed():
    """The cut is a property of the scores, not of the row order it was given."""

    tab = pd.DataFrame({"flow_in": [1.0, 1.0, 1.0, 1.0], "flow_out": [0.0, 0.0, 0.0, 0.8],
                        "absorbed": [0.3, 0.9, 0.6, 0.2], "term_score": [1.0, 1.0, 1.0, 0.2]},
                       index=["c", "a", "b", "z"])
    assert _lineages.top_terminals(tab, 2) == (["a", "b"], ["a", "b", "c"])
    # same scores, rows shuffled -> same answer
    assert _lineages.top_terminals(tab.iloc[[3, 0, 2, 1]], 2) == (["a", "b"], ["a", "b", "c"])
    # a cut below the tie is unambiguous and reports no tie
    assert _lineages.top_terminals(tab, 3) == (["a", "b", "c"], [])


def test_terminality_table_orders_ties_the_way_the_cut_does():
    """The printed ranking is the ranking n_lineages cuts, ties included."""
    a = _tied_flow(0.4, 0.6, names=("ta", "tb"))
    tab = dt.tl.terminality(a, cluster_key="clusters", source_clusters=["src"], show=False)
    assert list(map(str, tab.index)) == ["tb", "ta", "mid"]     # term_score desc, then absorbed desc
    # the value route reads the same frame, so both routes agree on the ORDER too
    dt.tl.lineages(a, ["src"], None, cluster_key="clusters", omega_term=0.5, verbose=False)
    by_value = list(map(str, a.uns["path_lineages"]))
    b = _tied_flow(0.4, 0.6, names=("ta", "tb"))
    dt.tl.lineages(b, ["src"], None, cluster_key="clusters", n_lineages=2, verbose=False)
    assert by_value == list(map(str, b.uns["path_lineages"])) == ["tb", "ta"]


def test_no_warning_when_the_cut_clears_the_tie():
    a = _tied_flow(0.6, 0.4)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", n_lineages=2,
                       verbose=False)
    assert not [w for w in rec if "tied at term_score" in str(w.message)]
    assert sorted(map(str, a.uns["path_lineages"])) == ["t1", "t2"]


def test_source_only_error_offers_both_routes():
    """Neither knob given -> the message must name omega_term AND n_lineages."""
    a = _ranked_flow()
    with pytest.raises(ValueError) as ei:
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", verbose=False)
    msg = str(ei.value)
    assert "omega_term=" in msg and "n_lineages=1" in msg      # the gap implies one


def test_suggested_n_lineages_agrees_with_the_suggested_omega_term():
    """The two routes offered in one sentence must not give different lineage counts.

    ``src -> {t1, t2}`` with no intermediate cluster: both sinks score exactly 1.0 and
    ``src`` is dropped, so EVERY gap in the ranking is zero and the gap index is no longer
    the number of clusters the suggested cutoff keeps.
    """
    obs = pd.DataFrame({"clusters": ["src", "t1", "t2"]}, index=["c0", "c1", "c2"])
    def bifurcation():
        a = ad.AnnData(X=np.zeros((3, 2), np.float32), obs=obs.copy())
        a.obs["pseudotime"] = [0.0, 1.0, 1.0]
        a.uns["dt_flow"] = {"edges": np.array([[0, 1], [0, 2]], np.int64),
                            "J": np.array([0.6, 0.4], float)}
        return a

    with pytest.raises(ValueError) as ei:
        dt.tl.lineages(bifurcation(), ["src"], None, cluster_key="clusters", verbose=False)
    msg = str(ei.value)
    omega = float(re.search(r"omega_term=([0-9.]+)", msg).group(1))
    n_sug = int(re.search(r"n_lineages=(\d+)", msg).group(1))

    by_value = bifurcation()
    dt.tl.lineages(by_value, ["src"], None, cluster_key="clusters", omega_term=omega,
                   verbose=False)
    by_count = bifurcation()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt.tl.lineages(by_count, ["src"], None, cluster_key="clusters", n_lineages=n_sug,
                       verbose=False)
    assert (sorted(map(str, by_count.uns["path_lineages"]))
            == sorted(map(str, by_value.uns["path_lineages"])) == ["t1", "t2"])


def test_no_count_route_is_suggested_when_nothing_is_rankable():
    """An empty ranking has no valid n_lineages, so the message must not invent one."""
    obs = pd.DataFrame({"clusters": ["src", "src", "src"]}, index=["c0", "c1", "c2"])
    a = ad.AnnData(X=np.zeros((3, 2), np.float32), obs=obs)
    a.obs["pseudotime"] = [0.0, 0.5, 1.0]
    a.uns["dt_flow"] = {"edges": np.array([[0, 1], [1, 2]], np.int64),
                        "J": np.array([1.0, 1.0], float)}
    with pytest.raises(ValueError, match="no terminals to rank") as ei:
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", verbose=False)
    assert "n_lineages=0" not in str(ei.value)


def test_terminality_mentions_the_count_route(capsys):
    dt.tl.terminality(_ranked_flow(), cluster_key="clusters", source_clusters=["src"])
    assert "n_lineages=" in capsys.readouterr().out


def test_trajectory_rejects_n_lineages_with_sinks_before_solving():
    """The guard runs up front, so a bare object never reaches endpoints."""
    a = _ranked_flow()
    with pytest.raises(ValueError, match="only applies in source-only mode"):
        dt.tl.trajectory(a, ["src"], ["hi"], cluster_key="clusters", n_lineages=1,
                         verbose=False)


@pytest.fixture(scope="module")
def graphed():
    # marker_noise=0 as in test_source_only: expression is monotone in t, so no mass flows
    # back into the source and the ranking is the clean sink-vs-mid comparison.
    return dt.pp.integrate(*line_toy(n=150, marker_noise=0.0),
                        alpha=0.9, p_keep=0.3, verbose=False)


def test_trajectory_forwards_n_lineages(graphed):
    obj = graphed.copy()
    dt.tl.trajectory(obj, ["source"], None, cluster_key="clusters", graph="intr",
                     basis="spatial", p_end=0.2, n_lineages=1, verbose=False)
    assert list(map(str, obj.uns["path_lineages"])) == ["sink"]
    assert np.isfinite(obj.obs["pseudotime"].to_numpy(float)).all()
