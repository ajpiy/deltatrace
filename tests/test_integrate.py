"""Pin deltatrace.pp.integrate to known input -> output."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from _line_toy import line_toy

import deltatrace as dt


def test_integrate_intersection_blend(sc_with_knn, wd_matrix):
    out = dt.pp.integrate(
        sc_with_knn,
        wd_matrix,
        alpha=0.5,
        p_keep=0.5,
        mode="intersection",
        verbose=False,
    )

    # Integrated graph stored under the default intersection graph name.
    assert "neighbors_intr_distances" in out.obsp
    assert "neighbors_intr_connectivities" in out.obsp
    assert out.uns["neighbors_intr"]["params"]["method"] == "intr"

    # p_keep=0.5 drops the two largest WD entries (the 10s), leaving edges
    # (0,1) and (1,2) — both present in the expression graph -> 4 directed
    # intersection edges.
    assert out.uns["neighbors_intr"]["params"]["n_edges_intersection"] == 4

    # The distances slot now holds the -ln(conn) transport cost
    # (clip(-ln(conn_sym), floor, cap)). The kNN side uses connectivities, not distances:
    #   conn(0,1) = 0.5*expr_c_n + 0.5*ext_c_n = 0.5*(0.5/0.8) + 0.5*(1 - 4/6) = 0.479166...
    #   conn(1,2) = 0.5*(0.8/0.8) + 0.5*(1 - 6/6) = 0.5
    # cap = q0.99 of -ln(conn) == -ln(conn(0,1)) (the max); floor = 0.5*min positive -> neither binds.
    conn01 = 0.5 * (0.5 / 0.8) + 0.5 * (1.0 - 4.0 / 6.0)
    conn12 = 0.5 * (0.8 / 0.8) + 0.5 * (1.0 - 6.0 / 6.0)
    dmat = out.obsp["neighbors_intr_distances"].tocsr()
    assert dmat[0, 1] == pytest.approx(-np.log(conn01), abs=1e-6)
    assert dmat[1, 2] == pytest.approx(-np.log(conn12), abs=1e-6)

    # connectivities slot still holds the blended [0, 1] connectivity.
    cmat = out.obsp["neighbors_intr_connectivities"].tocsr()
    assert cmat[0, 1] == pytest.approx(conn01, abs=1e-6)
    assert cmat[1, 2] == pytest.approx(conn12, abs=1e-6)

    # Every node has at least one edge -> no disconnected nodes.
    assert out.uns["disconnected_intr"]["n_disconnected"] == 0


def test_integrate_expression_mode_keeps_all_expr_edges(sc_with_knn, wd_matrix):
    out = dt.pp.integrate(
        sc_with_knn,
        wd_matrix,
        alpha=0.5,
        p_keep=0.5,
        mode="expression",
        verbose=False,
    )

    assert "neighbors_expr_distances" in out.obsp
    params = out.uns["neighbors_expr"]["params"]
    assert params["method"] == "expr"
    # All 4 directed expression edges are in the intersection here, so there are
    # no expression-only edges.
    assert params["n_edges_both"] == 4
    assert params["n_edges_expr_only"] == 0


def test_integrate_requires_knn_graph(wd_matrix):
    bare = ad.AnnData(
        X=np.zeros((3, 2)),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
    )
    with pytest.raises(ValueError, match="No kNN graph"):
        dt.pp.integrate(bare, wd_matrix, verbose=False)


# ------------------------------------------------------- obs_names alignment of the pair
def _reordered(dist, order):
    """The same distance object with its rows AND columns in ``order``."""
    names = list(dist.obs_names)
    perm = [names.index(n) for n in order]
    X = dist.X.toarray() if hasattr(dist.X, "toarray") else np.asarray(dist.X)
    return ad.AnnData(
        X=X[np.ix_(perm, perm)],
        obs=pd.DataFrame(index=list(order)),
        var=pd.DataFrame(index=list(order)),
    )


def _costs(out):
    return out.obsp["neighbors_intr_distances"].toarray()


def test_integrate_realigns_a_permuted_distance_matrix(sc_with_knn, wd_matrix):
    """A distance object in a different row order must be realigned, not fused blind.

    Nothing used to read obs_names at all, so the permuted matrix was accepted silently
    and every cell was regularized by another cell's spatial distances.
    """
    ref = dt.pp.integrate(sc_with_knn, wd_matrix, alpha=0.5, p_keep=0.5, verbose=False)
    shuffled = _reordered(wd_matrix, ["c2", "c0", "c1"])
    # The permutation must actually change the numbers, or the test proves nothing.
    assert not np.allclose(shuffled.X, wd_matrix.X.toarray())

    out = dt.pp.integrate(sc_with_knn, shuffled, alpha=0.5, p_keep=0.5, verbose=False)
    np.testing.assert_allclose(_costs(out), _costs(ref))
    assert list(out.obs_names) == list(sc_with_knn.obs_names)


def test_integrate_logs_the_realignment_at_level_2(sc_with_knn, wd_matrix, capsys):
    dt.pp.integrate(sc_with_knn, _reordered(wd_matrix, ["c2", "c0", "c1"]),
                    p_keep=0.5, verbose=2)
    assert "Realigned adata_dist" in capsys.readouterr().out


def test_integrate_rejects_a_distance_object_about_other_cells(sc_with_knn, wd_matrix):
    other = _reordered(wd_matrix, ["c0", "c1", "c2"])
    other.obs_names = ["x0", "x1", "c2"]
    other.var_names = ["x0", "x1", "c2"]
    with pytest.raises(ValueError, match="different cells"):
        dt.pp.integrate(sc_with_knn, other, p_keep=0.5, verbose=False)


def test_integrate_still_accepts_an_unlabeled_distance_matrix(sc_with_knn, wd_matrix):
    """A default 0,1,2,... index carries no identity, so it stays positional -- but the order
    is then unchecked, which the caller is told (naming the cells turns on the realignment)."""
    bare = ad.AnnData(X=wd_matrix.X.toarray())
    assert list(bare.obs_names) == ["0", "1", "2"]
    ref = dt.pp.integrate(sc_with_knn, wd_matrix, alpha=0.5, p_keep=0.5, verbose=False)
    with pytest.warns(UserWarning, match="no cell names") as rec:
        out = dt.pp.integrate(sc_with_knn, bare, alpha=0.5, p_keep=0.5, verbose=False)
    assert rec[0].filename == __file__
    np.testing.assert_allclose(_costs(out), _costs(ref))


def test_a_named_distance_matrix_is_realigned_silently(sc_with_knn, wd_matrix):
    import warnings
    shuffled = _reordered(wd_matrix, ["c2", "c0", "c1"])
    ref = dt.pp.integrate(sc_with_knn, wd_matrix, alpha=0.5, p_keep=0.5, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = dt.pp.integrate(sc_with_knn, shuffled, alpha=0.5, p_keep=0.5, verbose=False)
    np.testing.assert_allclose(_costs(out), _costs(ref))


# --- no spatial input: the plain expression graph ----------------------------------------

def test_without_a_distance_matrix_the_expression_graph_is_alpha_zero():
    """integrate(adata) with no adata_dist is exactly what mode='expression' at alpha=0
    builds from any distance matrix (that blend never reads the distances)."""
    adata, dist = line_toy(n=120)
    ref = dt.pp.integrate(adata, dist, alpha=0.0, mode="expression", verbose=0)
    out = dt.pp.integrate(adata, mode="expression", verbose=0)
    for kind in ("connectivities", "distances"):
        a, b = out.obsp[f"neighbors_expr_{kind}"], ref.obsp[f"neighbors_expr_{kind}"]
        assert (a != b).nnz == 0, kind
    params = out.uns["neighbors_expr"]["params"]
    assert params["alpha"] == 0.0 and params["spatial"] is False and params["p_keep"] is None
    # and it is the scanpy kNN graph itself, rescaled to a max of 1
    knn = adata.obsp["connectivities"]
    np.testing.assert_allclose(out.obsp["neighbors_expr_connectivities"].toarray(),
                               (knn / knn.max()).toarray())


def test_the_intersection_graph_needs_a_distance_matrix():
    adata, _ = line_toy(n=60)
    with pytest.raises(ValueError, match="needs adata_dist"):
        dt.pp.integrate(adata, verbose=0)                  # mode="intersection" by default
    with pytest.raises(ValueError, match="needs adata_dist"):
        dt.pp.graphs(adata, modes=("expression", "intersection"), verbose=0)


def test_graphs_and_trajectory_run_without_spatial_input():
    """pp.graphs(adata) builds just 'expr', and trajectory then runs on it by default."""
    adata, _ = line_toy(n=150)
    g = dt.pp.graphs(adata, verbose=0)
    assert [k for k in g.obsp if k.startswith("neighbors_")] == [
        "neighbors_expr_distances", "neighbors_expr_connectivities"]
    dt.tl.trajectory(g, ["source"], ["sink"], verbose=0)
    assert "endpoint_expr" in g.obs
    assert np.isfinite(g.obs["pseudotime"].to_numpy(float)).all()


def test_integrate_rejects_repeated_names_on_either_side(sc_with_knn, wd_matrix):
    """Duplicate labels leave no unambiguous permutation, so neither side may be guessed.

    AnnData only WARNS about non-unique obs_names, so a concatenated object that skipped
    obs_names_make_unique reaches here. Building a permutation from it copies one cell's
    spatial distances over another's and drops a third -- exactly the silent mis-fusion
    the alignment check exists to prevent, and worse than the old positional read.
    """
    dup_dist = _reordered(wd_matrix, ["c2", "c0", "c1"])
    dup_dist.obs_names = ["c0", "c0", "c1"]
    with pytest.raises(ValueError, match=r"adata_dist\.obs_names repeats"):
        dt.pp.integrate(sc_with_knn, dup_dist, p_keep=0.5, verbose=False)

    dup_sc = sc_with_knn.copy()
    dup_sc.obs_names = ["c0", "c0", "c2"]
    with pytest.raises(ValueError, match=r"adata_sc\.obs_names repeats"):
        dt.pp.integrate(dup_sc, _reordered(wd_matrix, ["c2", "c0", "c1"]),
                        p_keep=0.5, verbose=False)


def test_integrate_realigns_a_permuted_numeric_index(sc_with_knn, wd_matrix):
    """'0','1','2' out of order is a real, matchable set -- not a default index.

    A default index means "read me positionally", but that only holds while it IS the
    default order. Permuted numeric labels name the same cells as a numerically indexed
    adata_sc and the permutation is fully determined, so it must be applied.
    """
    num_sc = sc_with_knn.copy()
    num_sc.obs_names = ["0", "1", "2"]
    ref = dt.pp.integrate(num_sc, wd_matrix.copy(), alpha=0.5, p_keep=0.5, verbose=False)

    shuffled = _reordered(wd_matrix, ["c2", "c0", "c1"])
    shuffled.obs_names = shuffled.var_names = ["2", "0", "1"]
    assert not np.allclose(shuffled.X, wd_matrix.X.toarray())

    out = dt.pp.integrate(num_sc, shuffled, alpha=0.5, p_keep=0.5, verbose=False)
    np.testing.assert_allclose(_costs(out), _costs(ref))


def test_integrate_stays_positional_when_only_adata_sc_is_unlabeled(sc_with_knn,
                                                                     wd_matrix):
    """An unlabeled adata_sc has no identity to match real names against, so a named
    distance object about the same cells must still be read positionally, not rejected."""
    num_sc = sc_with_knn.copy()
    num_sc.obs_names = ["0", "1", "2"]
    ref = dt.pp.integrate(num_sc, wd_matrix.copy(), alpha=0.5, p_keep=0.5, verbose=False)
    named = _reordered(wd_matrix, ["c0", "c1", "c2"])       # real names, correct order
    out = dt.pp.integrate(num_sc, named, alpha=0.5, p_keep=0.5, verbose=False)
    np.testing.assert_allclose(_costs(out), _costs(ref))


def test_integrate_reports_shape_mismatches_as_valueerror(sc_with_knn, wd_matrix):
    """ValueError, not AssertionError: `python -O` strips asserts, and a non-square
    matrix used to die deep inside scipy instead."""
    short = ad.AnnData(X=wd_matrix.X.toarray()[:2])
    with pytest.raises(ValueError, match="Cell count mismatch"):
        dt.pp.integrate(sc_with_knn, short, p_keep=0.5, verbose=False)

    wide = ad.AnnData(X=np.ones((3, 4)))
    with pytest.raises(ValueError, match="must be square"):
        dt.pp.integrate(sc_with_knn, wide, p_keep=0.5, verbose=False)


# --- ingest guards: malformed priors must raise, well-formed ones must be untouched ----

def _swap_X(wd, X):
    return ad.AnnData(X=X, obs=wd.obs.copy(), var=wd.var.copy())


def _conn(out, graph="intr"):
    return out.obsp[f"neighbors_{graph}_connectivities"].toarray()


@pytest.mark.parametrize("bad,match", [(np.nan, "non-finite"), (np.inf, "non-finite"),
                                       (-1.0, "negative")])
def test_integrate_rejects_nonfinite_and_negative_distances(sc_with_knn, wd_matrix, bad, match):
    """These propagate silently: nan makes the p_keep quantile nan, which filters nothing
    and turns every stored value of both outputs nan without raising anywhere."""
    X = wd_matrix.X.copy().tolil().astype(float)
    X[0, 1] = bad
    with pytest.raises(ValueError, match=match):
        dt.pp.integrate(sc_with_knn, _swap_X(wd_matrix, X.tocsr()), verbose=False)


def test_integrate_ignores_a_stored_diagonal(sc_with_knn, wd_matrix):
    """A cell's distance to itself is never an edge, but it IS a stored value, so leaving
    it in drags the p_keep quantile down and silently retunes the filter."""
    ref = dt.pp.integrate(sc_with_knn, wd_matrix, p_keep=0.5, verbose=False)
    X = wd_matrix.X.copy().tolil().astype(float)
    X.setdiag(7.5)
    out = dt.pp.integrate(sc_with_knn, _swap_X(wd_matrix, X.tocsr()), p_keep=0.5, verbose=False)
    np.testing.assert_array_equal(_conn(out), _conn(ref))


def test_integrate_symmetrizes_a_half_filled_distance_matrix(sc_with_knn, wd_matrix):
    """The prior defines an undirected graph, so which triangle the caller filled in must
    not decide which edges survive the intersection."""
    import scipy.sparse as sp

    ref = dt.pp.integrate(sc_with_knn, wd_matrix, p_keep=0.5, verbose=False)
    upper = sp.triu(sp.csr_matrix(wd_matrix.X).astype(float), 1).tocsr()
    out = dt.pp.integrate(sc_with_knn, _swap_X(wd_matrix, upper), p_keep=0.5, verbose=False)
    np.testing.assert_array_equal(_conn(out), _conn(ref))


def test_integrate_reports_an_empty_graph_instead_of_an_index_error(sc_with_knn, wd_matrix):
    """At alpha=1.0 the connectivity is the prior's similarity alone, which is 0 on its
    longest kept edge -- so a single-distance prior empties the graph. That used to
    surface as a bare IndexError from inside np.quantile."""
    import scipy.sparse as sp

    X = sp.csr_matrix(wd_matrix.X).astype(float)
    flat = sp.csr_matrix((np.full(X.nnz, 3.0), X.indices, X.indptr), shape=X.shape)
    with pytest.raises(ValueError, match="no edges"):
        dt.pp.integrate(sc_with_knn, _swap_X(wd_matrix, flat), alpha=1.0, verbose=False)


def test_no_names_on_either_side_is_not_warned_about(sc_with_knn, wd_matrix):
    """Nothing to match when adata itself has a default index: stay positional, silently."""
    import warnings
    plain = sc_with_knn.copy()
    plain.obs_names = ["0", "1", "2"]
    bare = ad.AnnData(X=wd_matrix.X.toarray())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dt.pp.integrate(plain, bare, alpha=0.5, p_keep=0.5, verbose=False)


def test_unreachable_cells_are_told_to_hold_a_placeholder():
    """On a fragmented expression graph (no spatial input), cells no graph reaches get the
    LCC median pseudotime -- a placeholder the caller must hear about."""
    import scipy.sparse as sp_
    adata, _ = line_toy(n=150)
    island = np.arange(70, 76)
    for key in ("connectivities", "distances"):
        m = sp_.lil_matrix(adata.obsp[key])
        m[island, :] = 0
        m[:, island] = 0
        for a in island:                                   # the island keeps its own edges
            for b in island:
                if a != b:
                    m[a, b] = 1.0 if key == "connectivities" else 0.5
        adata.obsp[key] = m.tocsr()
    g = dt.pp.graphs(adata, verbose=0)
    with pytest.warns(UserWarning, match="placeholder") as rec:
        dt.tl.trajectory(g, ["source"], ["sink"], verbose=0)
    ours = [w for w in rec if "placeholder" in str(w.message)]
    assert len(ours) == 1 and ours[0].filename == __file__
    pt = g.obs["pseudotime"].to_numpy()
    assert np.unique(pt[island]).size == 1                 # the one median value
