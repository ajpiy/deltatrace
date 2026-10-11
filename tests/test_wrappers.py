"""pp.graphs and tl.trajectory — the one-call layer.

These pin the wrapper CONTRACT (both graphs land in one object; the composition
writes every downstream key), not the numerics of the stages, which are covered
by test_integrate / test_lineages and by notebook parity.
"""

import numpy as np
import pytest
from _line_toy import line_toy

import deltatrace as dt


@pytest.fixture(scope="module")
def toy():
    """Continuous source -> mid -> sink trajectory so the intersection graph stays
    connected (3 separated blobs would drop the sink out of the LCC)."""
    return line_toy(n=120)


def test_graphs_builds_both_into_one_object(toy):
    expr, dist = toy
    out = dt.pp.graphs(expr, dist, alpha=0.9, p_keep=0.3, verbose=False)

    for graph in ("expr", "intr"):
        assert f"neighbors_{graph}_distances" in out.obsp
        assert f"neighbors_{graph}_connectivities" in out.obsp
        assert f"neighbors_{graph}" in out.uns
    # the two graphs are genuinely different (intersection is a subset)
    assert out.obsp["neighbors_intr_connectivities"].nnz < out.obsp["neighbors_expr_connectivities"].nnz
    # intersection-family bookkeeping came across in the merge
    assert "disconnected_intr" in out.obs


def test_graphs_rejects_unknown_mode(toy):
    expr, dist = toy
    with pytest.raises(ValueError, match="unknown mode"):
        dt.pp.graphs(expr, dist, modes=("expression", "nope"), verbose=False)


def test_trajectory_composes_the_pipeline(toy):
    expr, dist = toy
    out = dt.pp.graphs(expr, dist, alpha=0.9, p_keep=0.3, verbose=False)

    res = dt.tl.trajectory(
        out, ["source"], ["sink"], cluster_key="clusters", graph="intr",
        basis="spatial", p_end=0.2, tau=100.0, seed=42, verbose=False,
    )

    assert set(res) == {"endpoints", "lineages"}
    # qrot wrote pseudotime for every cell
    pt = out.obs["pseudotime"].to_numpy(float)
    assert np.isfinite(pt).all()
    assert "dt_flow" in out.uns
    # velocity ran
    assert "dt_grid" in out.uns and "dt_raw_arrows" in out.uns
    # chain lineages wrote the canonical tradeSeq keys
    assert list(out.uns["path_lineages"]) == ["sink"]
    W = out.obsm["path_cellweights"]
    assert W.shape == (out.n_obs, 1)
    on = W.sum(1) > 1e-12
    np.testing.assert_allclose(W[on].sum(1), 1.0, atol=1e-6)
    for k in ("lineage", "lineage_confidence", "fate_leakage"):
        assert k in out.obs


def test_trajectory_source_only_ranks_the_fates_and_leaves_the_choice(toy, capsys):
    """Source-only without a cut runs the diffuse solve, prints the terminality ranking and
    returns it, and assigns no lineages: the caller picks the fates."""
    expr, dist = toy
    out = dt.pp.graphs(expr, dist, alpha=0.9, p_keep=0.3, verbose=False)
    with pytest.warns(UserWarning, match=r"no lineages assigned.*sink_clusters") as rec:
        res = dt.tl.trajectory(out, ["source"], None, cluster_key="clusters", graph="intr",
                               basis="spatial", p_end=0.2, verbose=1)
    assert rec[0].filename == __file__
    assert res["lineages"] is None
    assert res["terminality"].index[0] == "sink"            # the true fate ranks first
    assert "cluster terminality" in capsys.readouterr().out
    assert np.isfinite(out.obs["pseudotime"].to_numpy(float)).all()
    assert "lineage" not in out.obs


def test_trajectory_source_only_with_a_cut_assigns_lineages(toy):
    expr, dist = toy
    out = dt.pp.graphs(expr, dist, alpha=0.9, p_keep=0.3, verbose=False)
    res = dt.tl.trajectory(out, ["source"], None, cluster_key="clusters", graph="intr",
                           basis="spatial", p_end=0.2, n_lineages=1, verbose=False)
    assert set(res) == {"endpoints", "lineages"}
    assert list(out.uns["path_lineages"]) == ["sink"]


# --------------------------------------------------------------------------------------
# graph=None: "intr" when stored, else the only graph there is
# --------------------------------------------------------------------------------------
def test_trajectory_runs_on_a_lone_custom_graph(toy):
    expr, dist = toy
    g = dt.pp.integrate(expr, dist, mode="expression", key_added="mine", verbose=0)
    dt.tl.trajectory(g, ["source"], ["sink"], basis="spatial", verbose=0)
    assert "endpoint_mine" in g.obs


def test_trajectory_without_any_graph_points_to_pp_graphs(toy):
    expr, _ = toy
    with pytest.raises(ValueError, match=r"dt\.pp\.graphs\(adata\)"):
        dt.tl.trajectory(expr.copy(), ["source"], ["sink"], basis="spatial", verbose=0)


def test_trajectory_with_two_graphs_but_no_intr_asks_which(toy):
    expr, dist = toy
    g = dt.pp.graphs(expr, dist, modes=("expression",), verbose=0)
    extra = dt.pp.integrate(expr, dist, mode="expression", key_added="mine", verbose=0)
    for k in ("distances", "connectivities"):
        g.obsp[f"neighbors_mine_{k}"] = extra.obsp[f"neighbors_mine_{k}"]
    g.uns["neighbors_mine"] = extra.uns["neighbors_mine"]
    with pytest.raises(ValueError, match="Pass graph= explicitly"):
        dt.tl.trajectory(g, ["source"], ["sink"], basis="spatial", verbose=0)


# --------------------------------------------------------------------------------------
# source-only without a cut: namespaced runs and silence at verbose=0
# --------------------------------------------------------------------------------------
def test_source_only_ranks_its_own_run_and_is_quiet_at_verbose_0(toy, capsys):
    expr, dist = toy
    out = dt.pp.graphs(expr, dist, alpha=0.9, p_keep=0.3, verbose=False)
    dt.tl.trajectory(out, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                     basis="spatial", p_end=0.2, verbose=0)
    flat = dt.tl.terminality(out, cluster_key="clusters", source_clusters=["source"],
                             show=False)
    with pytest.warns(UserWarning, match="no lineages assigned") as rec:
        res = dt.tl.trajectory(out, ["source"], None, cluster_key="clusters", graph="intr",
                               basis="spatial", p_end=0.2, verbose=0, key_added="x")
    assert len([w for w in rec if "no lineages" in str(w.message)]) == 1
    assert "key_added='x'" in str(rec[0].message)
    assert "cluster terminality" not in capsys.readouterr().out
    ranked = dt.tl.terminality(out, cluster_key="clusters", source_clusters=["source"],
                               key_added="x", show=False)
    assert res["terminality"].equals(ranked)          # the source-only flow, not the flat one
    assert not res["terminality"].equals(flat)



# --------------------------------------------------------------------------------------
# input checks that fail early and say why
# --------------------------------------------------------------------------------------
@pytest.fixture
def graphed(toy):
    expr, dist = toy
    return dt.pp.graphs(expr, dist, alpha=0.9, p_keep=0.3, verbose=False)


def test_integer_labels_and_a_bare_string_label_are_accepted(graphed):
    """Leiden-style labels are strings in obs; [0] and "source" must still find them."""
    g = graphed.copy()
    g.obs["leiden"] = g.obs["clusters"].map({"source": "0", "mid": "1", "sink": "2"})
    dt.tl.trajectory(g, [0], [2], cluster_key="leiden", basis="spatial", verbose=0)
    assert set(g.obs["lineage"]) - {"unassigned"} == {"2"}
    h = graphed.copy()
    dt.tl.trajectory(h, "source", "sink", basis="spatial", verbose=0)
    assert "sink" in set(h.obs["lineage"])


def test_a_missing_cluster_column_names_the_fix(graphed):
    with pytest.raises(KeyError, match="cluster_key="):
        dt.tl.trajectory(graphed.copy(), ["source"], ["sink"], cluster_key="leiden",
                         basis="spatial", verbose=0)


def test_a_misspelt_label_is_reported_before_any_work(graphed):
    g = graphed.copy()
    with pytest.raises(ValueError, match=r"\['sorce'\] not found.*'source'"):
        dt.tl.trajectory(g, ["sorce"], ["sink"], basis="spatial", verbose=0)
    assert "endpoint_intr" not in g.obs


def test_an_unstored_graph_lists_the_stored_ones(graphed):
    with pytest.raises(ValueError, match=r"No graph 'intersection'.*\['expr', 'intr'\]"):
        dt.tl.trajectory(graphed.copy(), ["source"], ["sink"], graph="intersection",
                         basis="spatial", verbose=0)
    only_expr = dt.pp.graphs(*line_toy(n=60), modes=("expression",), verbose=0)
    with pytest.raises(ValueError, match="'intr' is built by"):
        dt.tl.trajectory(only_expr, ["source"], ["sink"], graph="intr", basis="spatial",
                         verbose=0)


def test_a_custom_neighbors_key_is_used_and_never_silently_replaced(toy):
    import scanpy as sc
    expr, dist = toy
    a = expr.copy()
    sc.pp.neighbors(a, n_neighbors=7, use_rep="X_pca", key_added="k7")
    g = dt.pp.graphs(a, dist, neighbors_key="k7", verbose=0)
    assert g.uns["neighbors_intr"]["params"]["n_neighbors"] == 7
    k7 = dt.pp.integrate(a, dist, alpha=0.0, mode="expression", neighbors_key="k7", verbose=0)
    knn = a.obsp["k7_connectivities"]
    np.testing.assert_allclose(k7.obsp["neighbors_expr_connectivities"].toarray(),
                               (knn / knn.max()).toarray())
    with pytest.raises(ValueError, match="neighbors_key='missing'"):
        dt.pp.graphs(a, dist, neighbors_key="missing", verbose=0)
