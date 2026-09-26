"""Source-only mode: no annotated sinks -> diffuse solve -> discover terminals.

A source->mid->sink toy has one obvious terminal ('sink'); the discovery must find it.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from _line_toy import line_toy

import deltatrace as dt
from deltatrace import _lineages


@pytest.fixture(scope="module")
def graphed():
    # marker_noise=0 keeps expression exactly monotone in t, so no mass flows back into
    # the source cluster and the terminality table is the clean sink-vs-mid comparison
    # these tests pin. See tests/_line_toy.py.
    return dt.pp.graphs(*line_toy(n=150, marker_noise=0.0),
                        alpha=0.9, p_keep=0.3, verbose=False)


def _diffuse(obj):
    gr = dt.tl.endpoints(obj, obj, ["source"], None, cluster_key="clusters", graph="intr",
                         p_end=0.2, sampling_mode="directed", verbose=False)
    dt.tl.qrot(gr, adata_basis=obj, adata_clust=obj, graph="intr", verbose=False)
    return gr


def test_endpoints_source_only_builds_diffuse_demand(graphed):
    """sink_clusters=None -> f sums to 0, sources negative, non-source cells absorb."""
    obj = graphed.copy()
    gr = dt.tl.endpoints(obj, obj, ["source"], None, cluster_key="clusters",
                         graph="intr", p_end=0.2, verbose=False)
    f = gr["f"]
    assert abs(f.sum()) < 1e-9                       # renormalized to 0
    assert (f < 0).any() and (f > 0).any()           # sources emit, absorbers take
    assert gr["sink_cells"] == []                    # no sinks sampled
    # every source cell is negative; absorbed mass is spread over many cells
    assert (f[gr["source_indices"]] < 0).all()
    assert int((f > 0).sum()) > len(gr["source_indices"])


def test_source_and_source_only_flows_differ(graphed):
    """The diffuse solve is genuinely different from a source+sink solve."""
    a = graphed.copy()
    gr = dt.tl.endpoints(a, a, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                         p_end=0.2, sampling_mode="directed", verbose=False)
    dt.tl.qrot(gr, adata_basis=a, adata_clust=a, graph="intr", verbose=False)
    pt_ss = a.obs["pseudotime"].to_numpy(float).copy()

    b = graphed.copy()
    _diffuse(b)
    pt_so = b.obs["pseudotime"].to_numpy(float)
    assert np.isfinite(pt_ss).all() and np.isfinite(pt_so).all()
    # different demand vectors -> not identical fields (mass now stops everywhere)
    assert not np.allclose(pt_ss, pt_so)


def test_terminality_ranks_sink_top_and_drops_pure_source(graphed):
    obj = graphed.copy()
    _diffuse(obj)
    tab = dt.tl.terminality(obj, cluster_key="clusters", show=False)
    assert "source" not in tab.index          # in==0 -> Term undefined -> dropped
    assert tab.index[0] == "sink"             # the true terminal ranks highest
    assert tab.loc["sink", "Term"] > tab.loc["mid", "Term"]


def test_suggest_omega_finds_the_clear_gap(graphed):
    obj = graphed.copy()
    _diffuse(obj)
    tab = dt.tl.terminality(obj, cluster_key="clusters", show=False)
    omega, info = _lineages.suggest_omega(tab["Term"].to_numpy())
    assert info["clear"]                                  # sink(1.0) vs mid(0.5): clear
    assert tab.loc["mid", "Term"] < omega < tab.loc["sink", "Term"]
    terminals = list(tab.index[tab["Term"] >= omega])
    assert terminals == ["sink"]


def test_lineages_discovers_the_terminal(graphed):
    obj = graphed.copy()
    _diffuse(obj)
    dt.tl.lineages(obj, ["source"], None, cluster_key="clusters", omega_term=0.75,
                   verbose=False)
    assert list(obj.uns["path_lineages"]) == ["sink"]
    W = obj.obsm["path_cellweights"]
    on = W.sum(1) > 1e-12
    np.testing.assert_allclose(W[on].sum(1), 1.0, atol=1e-6)


def test_no_terminal_above_omega_errors(graphed):
    obj = graphed.copy()
    _diffuse(obj)
    with pytest.raises(ValueError, match="No cluster has Term"):
        dt.tl.lineages(obj, ["source"], None, cluster_key="clusters", omega_term=2.0,
                       verbose=False)


def test_trajectory_source_only_end_to_end(graphed):
    obj = graphed.copy()
    res = dt.tl.trajectory(obj, ["source"], None, cluster_key="clusters", graph="intr",
                           basis="spatial", p_end=0.2, omega_term=0.75,
                           verbose=False)
    assert set(res) == {"endpoints", "lineages"}
    assert list(obj.uns["path_lineages"]) == ["sink"]
    assert np.isfinite(obj.obs["pseudotime"].to_numpy(float)).all()


def _unbalanced(graphed):
    """Relabel a handful of 'mid' cells into a tiny 'rare' cluster.

    line_toy splits the continuum into exact thirds, so uniform and
    cluster_balanced put the SAME demand on it and every test below would pass under
    either. The weighting only shows itself when the absorbing clusters differ in size.
    Labels are read at endpoint time and the graph does not depend on them, so relabelling
    after dt.pp.graphs is safe.
    """
    obj = graphed.copy()
    lab = obj.obs["clusters"].astype(str).to_numpy()
    lab[np.flatnonzero(lab == "mid")[-5:]] = "rare"
    obj.obs["clusters"] = pd.Categorical(lab)
    return obj


def _demand_by_cluster(gr, obj):
    """(cluster -> absorbed mass, cluster -> absorber count) over the cells with f > 0."""
    f = gr["f"]
    lab = obj.obs["clusters"].astype(str).to_numpy()
    pos = f > 0
    return ({c: float(f[pos & (lab == c)].sum()) for c in sorted(set(lab[pos]))},
            {c: int((pos & (lab == c)).sum()) for c in sorted(set(lab[pos]))})


def test_source_only_defaults_to_uniform(graphed):
    """The default must weight cells, not clusters: passing 'uniform' changes nothing."""
    obj = _unbalanced(graphed)
    kw = dict(cluster_key="clusters", graph="intr", p_end=0.2, verbose=False)
    default = dt.tl.endpoints(obj, obj, ["source"], None, **kw)["f"]
    uniform = dt.tl.endpoints(obj, obj, ["source"], None, sink_weights="uniform", **kw)["f"]
    np.testing.assert_array_equal(default, uniform)


def test_sink_weights_uniform_gives_every_absorber_equal_mass(graphed):
    obj = _unbalanced(graphed)
    gr = dt.tl.endpoints(obj, obj, ["source"], None, cluster_key="clusters", graph="intr",
                         p_end=0.2, sink_weights="uniform", verbose=False)
    f = gr["f"]
    assert abs(f.sum()) < 1e-9
    per_cell = f[f > 0]
    np.testing.assert_allclose(per_cell, per_cell[0], rtol=1e-9)

    # so a cluster's share of the mass is exactly its share of the absorbers
    mass, n = _demand_by_cluster(gr, obj)
    total = sum(n.values())
    for c in mass:
        assert mass[c] == pytest.approx(n[c] / total, rel=1e-9)


def test_sink_weights_cluster_balanced_equalises_clusters_not_cells(graphed):
    """The non-default: equal mass per cluster, which over-weights the tiny one per cell."""
    obj = _unbalanced(graphed)
    gr = dt.tl.endpoints(obj, obj, ["source"], None, cluster_key="clusters", graph="intr",
                         p_end=0.2, sink_weights="cluster_balanced", verbose=False)
    mass, n = _demand_by_cluster(gr, obj)
    assert set(mass) == {"mid", "rare", "sink"}
    for c in mass:
        assert mass[c] == pytest.approx(1.0 / len(mass), rel=1e-9)

    # 'rare' is the smallest cluster, so equal cluster mass means the most mass per cell
    per_cell = {c: mass[c] / n[c] for c in mass}
    assert per_cell["rare"] == max(per_cell.values())
    assert per_cell["rare"] > 5 * per_cell["sink"]


def test_the_two_weightings_actually_differ_when_clusters_are_unbalanced(graphed):
    obj = _unbalanced(graphed)
    kw = dict(cluster_key="clusters", graph="intr", p_end=0.2, verbose=False)
    u = dt.tl.endpoints(obj, obj, ["source"], None, sink_weights="uniform", **kw)
    b = dt.tl.endpoints(obj, obj, ["source"], None, sink_weights="cluster_balanced", **kw)

    # same absorber SET -- the weighting splits mass, it does not choose cells
    np.testing.assert_array_equal(u["f"] > 0, b["f"] > 0)
    assert not np.allclose(u["f"], b["f"])
    assert abs(u["f"].sum()) < 1e-9 and abs(b["f"].sum()) < 1e-9

    mass_u, _ = _demand_by_cluster(u, obj)
    mass_b, _ = _demand_by_cluster(b, obj)
    assert mass_b["rare"] > 4 * mass_u["rare"]      # tiny cluster inflated by the balancing
    assert mass_b["sink"] < mass_u["sink"]          # real terminal capped below its share


def test_sink_weights_rejects_unknown_value(graphed):
    with pytest.raises(ValueError, match="sink_weights must be"):
        dt.tl.endpoints(graphed.copy(), graphed, ["source"], None, cluster_key="clusters",
                        graph="intr", p_end=0.2, sink_weights="balanced", verbose=False)


def _backflux_flow():
    """src -> mid -> sink with a small mid -> src back-edge, so the source has non-zero
    inflow and Term_src = 1 - out/in = -9 (a large negative that hijacks the gap search)."""
    obs = pd.DataFrame({"clusters": ["src", "mid", "sink"]}, index=["c0", "c1", "c2"])
    a = ad.AnnData(X=np.zeros((3, 2), np.float32), obs=obs)
    a.obs["pseudotime"] = [0.0, 0.5, 1.0]
    a.uns["dt_flow"] = {
        "edges": np.array([[0, 1], [1, 2], [1, 0]], np.int64),
        "J": np.array([1.0, 0.9, 0.1], float),
    }
    return a


def test_terminality_source_exclusion_saves_the_gap_search():
    """Without the source drop, Term_src=-9 opens the largest gap and suggest_omega falsely
    declares the interior cluster terminal; passing source_clusters recovers the true sink."""
    a = _backflux_flow()
    raw = dt.tl.terminality(a, cluster_key="clusters", show=False)
    assert "src" in raw.index and raw.loc["src", "Term"] < -1        # the poison row
    omega_bad, _ = _lineages.suggest_omega(raw["Term"].to_numpy())
    assert "mid" in list(raw.index[raw["Term"] >= omega_bad])        # bug reproduced

    tab = dt.tl.terminality(a, cluster_key="clusters", source_clusters=["src"], show=False)
    assert "src" not in tab.index                                    # source dropped
    omega, _ = _lineages.suggest_omega(tab["Term"].to_numpy())
    assert list(tab.index[tab["Term"] >= omega]) == ["sink"]         # fixed


def test_source_only_lineages_ignore_backflux_source():
    """The source-only lineages path passes source_clusters through, so a back-flux source
    never surfaces as a discovered terminal."""
    a = _backflux_flow()
    dt.tl.lineages(a, ["src"], None, cluster_key="clusters", omega_term=0.5, verbose=False)
    assert list(a.uns["path_lineages"]) == ["sink"]
