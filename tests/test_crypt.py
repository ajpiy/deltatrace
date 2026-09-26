"""datasets.crypt — the crypt toy keeps its promise at package defaults.

The promise is geometric: at p_keep=0.32 the intersection graph cuts every expression
link across the lumen but keeps the two walls joined through the base, so ``intr`` is ONE
connected U, every wall cell reaches its own top, and no flow crosses the lumen, while an
expression-only graph still sends about a third of the wall cells to the wrong top.
"""

import contextlib
import io

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

import deltatrace as dt

SOURCES, SINKS = ["progenitor"], ["fate_A", "fate_B"]


def _intr_components(g):
    C = g.obsp[g.uns["neighbors_intr"]["connectivities_key"]]
    return connected_components(C, directed=False)


def _fates(g, key, adata):
    """(share of wall cells sent to the wrong top, share of flow on edges joining the walls)."""
    tf = adata.obs["true_fate"].astype(str).to_numpy()
    wall = adata.obs["wall"].astype(str).to_numpy()
    fated = np.isin(tf, SINKS)
    lin = g.obs[f"lineage_{key}"].astype(str).to_numpy()
    fl = g.uns[f"dt_flow_{key}"]
    J = np.asarray(fl["J"], float).ravel()
    E = np.asarray(fl["edges"], np.int64).reshape(-1, 2)
    u, v = E[:, 0], E[:, 1]
    cross = (wall[u] != wall[v]) & (wall[u] != "base") & (wall[v] != "base")
    called = lin != "unassigned"  # wall tips beyond the centered sink cells get no flow
    return float((called & (lin != tf))[fated].mean()), float(J[cross].sum() / J.sum())


def test_fields_and_units():
    adata, adata_dist = dt.datasets.crypt(n=80)
    assert adata.shape == (80, 8)
    assert list(adata.var_names[:4]) == ["stem", "mature", "marker_A", "marker_B"]
    assert (adata.X >= 0).all()
    assert list(adata.obs["clusters"].cat.categories) == ["progenitor", "transit", "fate_A", "fate_B"]
    assert set(adata.obs["wall"].astype(str)) == {"base", "left", "right"}
    # the base is exactly the progenitors; each fate sits on one wall
    base = adata.obs["wall"] == "base"
    assert (base == (adata.obs["clusters"] == "progenitor")).all()
    fate_wall = adata.obs.groupby("clusters", observed=True)["wall"].agg(lambda w: set(w.astype(str)))
    assert fate_wall["fate_A"] == {"left"} and fate_wall["fate_B"] == {"right"}
    assert adata.obsm["X_expr2d"].shape == (80, 2) and "X_pca" in adata.obsm
    assert adata.uns["story"] == {"source_clusters": SOURCES, "sink_clusters": SINKS,
                                  "decoy_clusters": []}
    assert adata.uns["crypt"]["units"] == "tissue units"
    assert list(adata_dist.obs_names) == list(adata.obs_names) == list(adata_dist.var_names)
    xy = adata.obsm["spatial"]
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.testing.assert_allclose(adata_dist.X, D, atol=1e-9)


def test_deterministic_in_seed():
    a1, d1 = dt.datasets.crypt(n=80, seed=3)
    a2, d2 = dt.datasets.crypt(n=80, seed=3)
    np.testing.assert_array_equal(a1.X, a2.X)
    np.testing.assert_array_equal(d1.X, d2.X)
    assert (a1.obsp["connectivities"] != a2.obsp["connectivities"]).nnz == 0
    a3, _ = dt.datasets.crypt(n=80, seed=4)
    assert not np.array_equal(a1.X, a3.X)


def test_cells_are_not_stored_wall_by_wall():
    # downstream tie-breaks take the lowest index; build order would favor the left wall
    adata, _ = dt.datasets.crypt(n=150)
    first = adata.obs["wall"].astype(str).to_numpy()[:40]
    assert {"left", "right"} <= set(first)


@pytest.mark.parametrize(("n", "seed"), [(n, s) for n in (80, 150) for s in range(5)] + [(1000, 7)])
def test_intr_is_one_component_and_the_lumen_is_cut(n, seed):
    adata, adata_dist = dt.datasets.crypt(n=n, seed=seed)
    g = dt.pp.graphs(adata, adata_dist, verbose=0)
    ncomp, _ = _intr_components(g)
    assert ncomp == 1
    with contextlib.redirect_stdout(io.StringIO()):
        dt.tl.trajectory(g, SOURCES, SINKS, key_added="i", verbose=0)
    wrong, cross = _fates(g, "i", adata)
    assert wrong == 0.0
    assert cross == 0.0


def test_expression_alone_cannot_tell_the_walls_apart():
    adata, adata_dist = dt.datasets.crypt(n=150)
    g = dt.pp.integrate(adata, adata_dist, alpha=0.0, mode="expression", verbose=0)
    with contextlib.redirect_stdout(io.StringIO()):
        dt.tl.trajectory(g, SOURCES, SINKS, key_added="e", verbose=0)
    wrong, cross = _fates(g, "e", adata)
    assert wrong > 0.2
    assert cross > 0.2


def test_false_fate_is_cut_off_as_its_own_component():
    adata, adata_dist = dt.datasets.crypt(n=150, false_fate=0.03)
    false = (adata.obs["clusters"] == "false_fate").to_numpy()
    assert false.sum() >= 3
    assert adata.uns["story"]["decoy_clusters"] == ["false_fate"]
    assert np.isnan(adata.obs["true_time"].to_numpy()[false]).all()
    g = dt.pp.graphs(adata, adata_dist, verbose=0)
    _, comp = _intr_components(g)
    main = np.argmax(np.bincount(comp))
    np.testing.assert_array_equal(comp != main, false)


def test_false_fate_warns_below_150_cells():
    with pytest.warns(UserWarning, match="false-fate"):
        dt.datasets.crypt(n=80, false_fate=0.05)


@pytest.mark.parametrize("bad", [dict(n=30), dict(fork_start=1.5), dict(lumen_width=0),
                                 dict(false_fate=0.6), dict(expr_noise=-0.1), dict(n_neighbors=1)])
def test_rejects_bad_arguments(bad):
    with pytest.raises(ValueError, match=r"crypt\(\)"):
        dt.datasets.crypt(**bad)


@pytest.mark.parametrize("bad", [dict(n=150.0), dict(n_neighbors=20.5)])
def test_rejects_non_integer_counts(bad):
    with pytest.raises(TypeError, match=r"crypt\(\)"):
        dt.datasets.crypt(**bad)
