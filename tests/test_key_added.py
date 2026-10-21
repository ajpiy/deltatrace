"""key_added namespacing: two runs in ONE object without clobbering each other.

This is what the expression-vs-intersection comparison needs — previously the second
trajectory() overwrote the first's obs['pseudotime'].
"""

import numpy as np
import pytest
from _line_toy import line_toy

import deltatrace as dt

RUN_KEYS_OBS = ("pseudotime", "pseudotime_raw", "lineage", "lineage_confidence",
                "fate_leakage")
RUN_KEYS_OBSM = ("path_cellweights", "path_pseudotime", "path_flowmass")
RUN_KEYS_UNS = ("dt_flow", "dt_grid", "dt_raw_arrows", "path_lineages")


@pytest.fixture(scope="module")
def graphed():
    return dt.pp.integrate(*line_toy(n=120, marker_noise=0.0),
                        alpha=0.9, p_keep=0.3, verbose=False)


def _run(obj, graph, key_added):
    return dt.tl.trajectory(
        obj, ["source"], ["sink"], cluster_key="clusters", graph=graph, basis="spatial",
        p_end=0.2, verbose=False, key_added=key_added,
    )


def test_two_runs_coexist_in_one_object(graphed):
    obj = graphed.copy()
    _run(obj, "expr", "expr")
    _run(obj, "intr", "intr")

    for k in RUN_KEYS_OBS:
        assert f"{k}_expr" in obj.obs and f"{k}_intr" in obj.obs
        assert k not in obj.obs, f"flat '{k}' must not be written when key_added is given"
    for k in RUN_KEYS_OBSM:
        assert f"{k}_expr" in obj.obsm and f"{k}_intr" in obj.obsm
        assert k not in obj.obsm
    for k in RUN_KEYS_UNS:
        assert f"{k}_expr" in obj.uns and f"{k}_intr" in obj.uns
        assert k not in obj.uns


def test_second_run_does_not_clobber_the_first(graphed):
    """The whole point of key_added: a later run must leave earlier keys untouched.

    (On this toy expr and intr are the same graph, so we test INDEPENDENCE of the
    stored keys, not that the numbers differ — identical numbers here are correct.)
    """
    obj = graphed.copy()
    _run(obj, "intr", "intr")
    first_pt = obj.obs["pseudotime_intr"].to_numpy(float).copy()
    first_W = obj.obsm["path_cellweights_intr"].copy()
    assert np.isfinite(first_pt).all()

    _run(obj, "expr", "expr")           # second run on the SAME object

    np.testing.assert_array_equal(obj.obs["pseudotime_intr"].to_numpy(float), first_pt)
    np.testing.assert_array_equal(obj.obsm["path_cellweights_intr"], first_W)
    assert "pseudotime_expr" in obj.obs and "path_cellweights_expr" in obj.obsm


def test_each_run_reads_its_own_flow(graphed):
    """qrot(key_added='k') must write dt_flow_k, and lineages(key_added='k') must read
    THAT flow — not a stale default dt_flow."""
    obj = graphed.copy()
    _run(obj, "intr", "intr")
    # no un-suffixed dt_flow should exist to accidentally fall back on
    assert "dt_flow" not in obj.uns
    # lineages consumed dt_flow_intr (same edge count as qrot wrote)
    assert obj.uns["dt_flow_intr"]["edges"].shape[0] == obj.uns["dt_flow_intr"]["J"].shape[0]
    assert obj.obsm["path_cellweights_intr"].shape[0] == obj.n_obs


def test_graph_keys_are_not_suffixed(graphed):
    """neighbors_<graph>/disconnected_<graph> are scoped by graph, not by run."""
    obj = graphed.copy()
    _run(obj, "intr", "intr")
    assert "neighbors_intr_distances" in obj.obsp
    assert "neighbors_intr_intr_distances" not in obj.obsp
    assert "disconnected_intr" in obj.obs


def test_default_writes_legacy_flat_keys(graphed):
    """key_added=None must keep the R/tradeSeq-facing names unchanged."""
    obj = graphed.copy()
    _run(obj, "intr", None)
    for k in RUN_KEYS_OBS:
        assert k in obj.obs
    for k in RUN_KEYS_OBSM:
        assert k in obj.obsm
    for k in RUN_KEYS_UNS:
        assert k in obj.uns


def test_stagewise_calls_accept_key_added(graphed):
    """The stages must be drivable by hand with the same namespacing."""
    obj = graphed.copy()
    gr = dt.tl.endpoints(obj, ["source"], ["sink"], cluster_key="clusters",
                         graph="intr", p_end=0.2, verbose=False)
    dt.tl.qrot(gr, adata=obj, graph="intr", verbose=False,
               key_added="hand")
    dt.tl.compute_velocity_grid(obj, basis="spatial", verbose=False, key_added="hand")
    dt.tl.lineages(obj, ["source"], ["sink"], cluster_key="clusters", verbose=False,
                   key_added="hand")
    assert "pseudotime_hand" in obj.obs
    assert "dt_grid_hand" in obj.uns
    assert "path_cellweights_hand" in obj.obsm


# ----------------------------------------- the two entry points that had no key_added yet
def test_widest_path_accepts_key_added(graphed):
    """tl.widest_path was the only tl entry point that could not see a namespaced run."""
    obj = graphed.copy()
    res = _run(obj, "intr", "intr")
    src = res["endpoints"]["source_indices"][0]
    dst = res["endpoints"]["sink_indices"][0]

    path, cap = dt.tl.widest_path(obj, src, dst, key_added="intr")
    assert path[0] == src and path[-1] == dst
    assert cap > 0


def test_widest_path_reports_a_missing_flow_like_its_siblings(graphed):
    """A bare adata.uns[flow_key] lookup raised a naked KeyError; the siblings raise
    ValueError naming the key and the call that writes it."""
    obj = graphed.copy()
    _run(obj, "intr", "intr")
    assert "dt_flow" not in obj.uns                      # nothing to fall back on

    with pytest.raises(ValueError, match=r"dt_flow"):
        dt.tl.widest_path(obj, 0, 1)
    with pytest.raises(ValueError, match=r"dt_flow_nope"):
        dt.tl.widest_path(obj, 0, 1, key_added="nope")


def test_pl_pseudotime_can_plot_a_namespaced_run(graphed):
    """A key_added run could be computed but not plotted: pl.pseudotime hardcoded
    uns['dt_grid'] and obs['pseudotime']."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj = graphed.copy()
    _run(obj, "intr", "intr")
    try:
        _, ax = dt.pl.pseudotime(obj, key_added="intr", show=False)
        # basis came from dt_grid_intr, colors from pseudotime_intr
        np.testing.assert_allclose(
            np.asarray(ax.collections[0].get_array(), float),
            obj.obs["pseudotime_intr"].to_numpy(float),
        )
        with pytest.raises(ValueError, match=r"dt_grid_nope"):
            dt.pl.pseudotime(obj, key_added="nope", show=False)
    finally:
        plt.close("all")
