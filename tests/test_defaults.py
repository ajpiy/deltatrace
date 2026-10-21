"""The public defaults are the benchmarked configuration, and ``tl.trajectory`` forwards the
same values the stages use when called by hand (``basis`` aside: only ``trajectory`` picks
its embedding itself).

Benchmarked for the intersection graph (scRNA-seq with a paired ST section): alpha 0.3,
p_keep 0.3, p_end 0.15, p_basis "cluster", centered sampling, tau "auto" (0.3 x cells in the largest
connected piece).
"""

import inspect
import warnings

import numpy as np
import pytest
from _line_toy import line_toy

import deltatrace as dt


def _defaults(fn, names):
    params = inspect.signature(fn).parameters
    return {n: params[n].default for n in names}


def test_graph_defaults_are_the_benchmarked_ones():
    for fn in (dt.pp.integrate,):
        assert _defaults(fn, ["alpha", "p_keep"]) == {"alpha": 0.3, "p_keep": 0.3}


def test_endpoint_defaults_are_the_benchmarked_ones():
    want = {"p_end": 0.15, "p_basis": "cluster", "sampling_mode": "centered",
            "cluster_key": "clusters"}
    assert _defaults(dt.tl.endpoints, want) == want
    assert _defaults(dt.tl.trajectory, want) == want


def test_trajectory_forwards_the_stage_defaults_unchanged():
    """Every argument trajectory only forwards defaults to what the stage itself uses."""
    to_endpoints = ["p_end", "p_basis", "alloc_power", "sampling_mode", "directed_metric",
                    "seed", "sink_weights", "cluster_key"]
    assert (_defaults(dt.tl.trajectory, to_endpoints)
            == _defaults(dt.tl.endpoints, to_endpoints))
    assert (_defaults(dt.tl.trajectory, ["tau", "reconnect_graph"])
            == _defaults(dt.tl.qrot, ["tau", "reconnect_graph"]))
    assert _defaults(dt.tl.lineages, ["cluster_key"]) == {"cluster_key": "clusters"}


def test_qrot_defaults_to_auto_tau():
    assert _defaults(dt.tl.qrot, ["tau"]) == {"tau": "auto"}
    assert _defaults(dt.tl.qrot_graph, ["tau"]) == {"tau": 100}


def test_wasserstein_uses_every_core_by_default():
    assert _defaults(dt.pp.wasserstein, ["n_jobs"]) == {"n_jobs": -1}


def test_the_readme_call_runs_on_the_toy_at_defaults():
    """pp.integrate -> tl.trajectory -> pl.pseudotime with nothing but the required
    arguments: the basis is found (obsm['spatial']) and every declared fate is reached."""
    import matplotlib
    matplotlib.use("Agg")
    g = dt.pp.integrate(*line_toy(n=150), verbose=0)
    dt.tl.trajectory(g, ["source"], ["sink"], verbose=0)
    assert g.uns["dt_grid"]["basis"] == "spatial"
    assert np.isfinite(g.obs["pseudotime"].to_numpy(float)).all()
    assert "sink" in set(g.obs["lineage"])
    dt.pl.pseudotime(g, show=False)


# --------------------------------------------------------------------------------------
# sampling_mode: a dict that leaves a role out
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy_graph():
    return dt.pp.integrate(*line_toy(n=150), alpha=0.5, p_keep=0.1, verbose=0)


def test_a_sampling_dict_with_an_unknown_role_is_an_error(toy_graph):
    a = toy_graph.copy()
    with pytest.raises(ValueError, match="'source' and/or 'sink'"):
        dt.tl.endpoints(a, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                        sampling_mode={"sinks": "random"}, verbose=0)


def test_a_role_missing_from_a_sampling_dict_is_centered(toy_graph):
    kw = dict(cluster_key="clusters", graph="intr", p_end=0.10, seed=5, verbose=0)
    a, b = toy_graph.copy(), toy_graph.copy()
    full = dt.tl.endpoints(a, ["source"], ["sink"], sampling_mode="centered", **kw)
    part = dt.tl.endpoints(b, ["source"], ["sink"], sampling_mode={}, **kw)
    assert full["source_cells"] == part["source_cells"]
    assert full["sink_cells"] == part["sink_cells"]


# --------------------------------------------------------------------------------------
# p_basis="lcc" can hand a declared cluster no cells: that must be said
# --------------------------------------------------------------------------------------
def test_a_zero_sink_quota_warns_and_names_the_way_out(toy_graph):
    a = toy_graph.copy()
    with pytest.warns(UserWarning, match="will have empty lineages") as rec:
        gr = dt.tl.endpoints(a, ["source"], ["mid", "sink"], cluster_key="clusters",
                             graph="intr", p_end=0.005, seed=42, verbose=0, p_basis="lcc")
    msg = str(rec[0].message)
    assert "p_basis='cluster'" in msg
    assert rec[0].filename == __file__                 # points at the caller's line
    assert len(set(gr["sink_cluster_map"].values())) == 1   # one sink cluster really is empty


def test_a_zero_quota_warning_points_at_the_caller_through_trajectory(toy_graph):
    a = toy_graph.copy()
    with pytest.warns(UserWarning, match="will have empty lineages") as rec:
        dt.tl.trajectory(a, ["source"], ["mid", "sink"], graph="intr", p_end=0.005,
                         p_basis="lcc", verbose=0)
    assert [w.filename for w in rec if "get none" in str(w.message)] == [__file__]


def test_a_zero_source_quota_warns_without_lineage_talk(toy_graph):
    a = toy_graph.copy()
    with pytest.warns(UserWarning, match=r"get none\. Raise p_end") as rec:
        dt.tl.endpoints(a, ["source", "mid"], ["sink"], cluster_key="clusters",
                        graph="intr", p_end=0.005, seed=42, verbose=0, p_basis="lcc")
    assert "source cluster" in str(rec[0].message)
    assert "lineages" not in str(rec[0].message)


def test_the_cluster_basis_never_leaves_a_cluster_empty(toy_graph):
    a = toy_graph.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gr = dt.tl.endpoints(a, ["source"], ["mid", "sink"], cluster_key="clusters",
                             graph="intr", p_end=0.005, seed=42, verbose=0,
                             p_basis="cluster")
    assert set(gr["sink_cluster_map"].values()) == {"mid", "sink"}


# --------------------------------------------------------------------------------------
# trajectory(basis=None): tissue first, then UMAP, checked before any solve
# --------------------------------------------------------------------------------------
def test_basis_prefers_the_tissue_then_umap(toy_graph):
    a = toy_graph.copy()
    a.obsm["X_umap"] = a.obsm["X_pca"][:, :2]
    dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", verbose=0)
    assert a.uns["dt_grid"]["basis"] == "spatial"

    b = toy_graph.copy()
    b.obsm["X_umap"] = b.obsm["X_pca"][:, :2]
    del b.obsm["spatial"]
    dt.tl.trajectory(b, ["source"], ["sink"], graph="intr", verbose=0)
    assert b.uns["dt_grid"]["basis"] == "umap"

    c = toy_graph.copy()                               # the tissue stored scanpy-style
    c.obsm["X_umap"] = c.obsm["X_pca"][:, :2]
    c.obsm["X_spatial"] = c.obsm.pop("spatial")
    dt.tl.trajectory(c, ["source"], ["sink"], graph="intr", verbose=0)
    assert c.uns["dt_grid"]["basis"] == "spatial"
    np.testing.assert_array_equal(c.uns["dt_raw_arrows"]["X"], c.obsm["X_spatial"][:, 0])


def _bad_tissue(obj, kind):
    xy = obj.obsm["spatial"]
    if kind == "3-D":
        return np.column_stack([xy, np.zeros(len(xy))])
    bad = xy.astype(float).copy()
    bad[0] = np.nan
    return bad


@pytest.mark.parametrize("kind", ["3-D", "NaN"])
def test_an_unusable_tissue_embedding_falls_back_to_umap(toy_graph, kind):
    """The tissue coordinates are preferred only when they can carry the 2-D grid: a
    3-D or non-finite obsm['spatial'] must not crash after the solve."""
    a = toy_graph.copy()
    a.obsm["X_umap"] = a.obsm["X_pca"][:, :2]
    a.obsm["spatial"] = _bad_tissue(a, kind)
    with pytest.warns(UserWarning, match="Drawing the velocity grid on 'umap'"):
        dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", verbose=0)
    assert a.uns["dt_grid"]["basis"] == "umap"


@pytest.mark.parametrize("kind", ["3-D", "NaN"])
def test_an_unusable_embedding_fails_before_the_solve(toy_graph, kind):
    a = toy_graph.copy()
    a.obsm["spatial"] = _bad_tissue(a, kind)
    for basis in ("spatial", None):                   # explicit, and nothing to fall back to
        with pytest.raises(ValueError, match="finite 2-D"):
            dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", basis=basis, verbose=0)
    assert "endpoint_intr" not in a.obs and "pseudotime" not in a.obs


def test_a_dataframe_embedding_draws_the_grid(toy_graph):
    import pandas as pd
    a = toy_graph.copy()
    a.obsm["spatial"] = pd.DataFrame(a.obsm["spatial"], index=a.obs_names, columns=["x", "y"])
    dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", verbose=0)
    assert a.uns["dt_grid"]["basis"] == "spatial"


@pytest.mark.parametrize("basis", [None, "tsne"])
def test_a_missing_embedding_fails_before_the_solve(toy_graph, basis):
    a = toy_graph.copy()
    del a.obsm["spatial"]
    with pytest.raises(KeyError):
        dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", basis=basis, verbose=0)
    assert "endpoint_intr" not in a.obs and "pseudotime" not in a.obs   # nothing ran


def test_no_embedding_is_fine_without_the_velocity_step(toy_graph):
    a = toy_graph.copy()
    del a.obsm["spatial"]
    dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", compute_velocity=False,
                     verbose=0)
    assert "dt_grid" not in a.uns and "pseudotime" in a.obs
