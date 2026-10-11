"""Pin deltatrace.pp.wasserstein to known input -> output."""

import warnings

import numpy as np
import pytest
from scipy.spatial.distance import squareform

import deltatrace as dt


def _square(ad_wd):
    return squareform(ad_wd.uns["distances"])


@pytest.mark.parametrize("n_jobs", [1, -1], ids=["serial", "parallel"])
def test_wasserstein_known_distances(mapping_and_spatial, n_jobs):
    M, adata_sp = mapping_and_spatial

    ad_wd = dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=n_jobs)

    # Indexed by the cells of M, distances condensed in pdist order: (0,1), (0,2), (1,2).
    assert ad_wd.shape == (3, 0)
    assert list(ad_wd.obs_names) == ["c0", "c1", "c2"]
    dc = ad_wd.uns["distances"]
    assert dc.dtype == np.float64 and dc.shape == (3,)
    np.testing.assert_allclose(dc, [3.0, 0.0, 3.0], atol=1e-12)

    WD = _square(ad_wd)
    expected = np.array(
        [[0.0, 3.0, 0.0], [3.0, 0.0, 3.0], [0.0, 3.0, 0.0]],
    )
    assert np.allclose(WD, expected, atol=1e-5)

    # Symmetric, zero diagonal.
    assert np.allclose(WD, WD.T)
    assert np.allclose(np.diag(WD), 0.0)

    # Stored metadata.
    assert ad_wd.uns["cost"] == "euclidean"
    assert ad_wd.uns["mass_threshold"] == pytest.approx(0.9)
    assert ad_wd.uns["d_max"] == pytest.approx(3.0, abs=1e-5)


@pytest.mark.parametrize("n_jobs", [1, -1], ids=["serial", "parallel"])
def test_wasserstein_pxl_to_um_scales_distances(mapping_and_spatial, n_jobs):
    M, adata_sp = mapping_and_spatial

    ad_wd = dt.pp.wasserstein(M, adata_sp, pxl_to_um=2.0, verbose=False, n_jobs=n_jobs)
    WD = _square(ad_wd)

    # Distances scale linearly with the coordinate scale.
    assert WD[0, 1] == pytest.approx(6.0, abs=1e-5)


def test_wasserstein_requires_spatial(mapping_and_spatial):
    M, adata_sp = mapping_and_spatial
    del adata_sp.obsm["spatial"]

    with pytest.raises(KeyError, match="spatial"):
        dt.pp.wasserstein(M, adata_sp, verbose=False)


def test_wasserstein_spot_count_mismatch(mapping_and_spatial):
    M, adata_sp = mapping_and_spatial
    adata_sp = adata_sp[:2].copy()  # 2 spots, but M has 3 spot columns

    with pytest.raises(ValueError, match="spot"):
        dt.pp.wasserstein(M, adata_sp, verbose=False)


def test_serial_and_parallel_agree_exactly():
    """n_jobs only sets speed, on a mapping with real spread."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(0)
    n_cells, n_spots = 25, 40
    X = rng.random((n_cells, n_spots)) ** 4
    X /= X.sum(1, keepdims=True)
    spots = pd.DataFrame(index=[f"s{i}" for i in range(n_spots)])
    M = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]), var=spots)
    sp_ = ad.AnnData(X=np.zeros((n_spots, 1)), obs=spots)
    sp_.obsm["spatial"] = rng.random((n_spots, 2)) * 500
    a = dt.pp.wasserstein(M, sp_, verbose=False, n_jobs=1).uns["distances"]
    b = dt.pp.wasserstein(M, sp_, verbose=False, n_jobs=-1).uns["distances"]
    np.testing.assert_array_equal(a, b)


def test_spots_are_matched_by_name_not_order(mapping_and_spatial):
    """A reordered spatial object (same spots, other order) gives the same distances: the
    mapping's spot columns are matched to the spots by name."""
    M, adata_sp = mapping_and_spatial
    ref = dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=1).uns["distances"]
    shuffled = adata_sp[["s2", "s0", "s1"]].copy()
    out = dt.pp.wasserstein(M, shuffled, verbose=False, n_jobs=1).uns["distances"]
    np.testing.assert_allclose(out, ref, atol=1e-6)


def test_spots_the_mapping_does_not_know_are_an_error(mapping_and_spatial):
    M, adata_sp = mapping_and_spatial
    other = adata_sp.copy()
    other.obs_names = ["s0", "s1", "zz"]
    with pytest.raises(ValueError, match="different spots"):
        dt.pp.wasserstein(M, other, verbose=False, n_jobs=1)


def test_repeated_spot_names_are_an_error(mapping_and_spatial):
    M, adata_sp = mapping_and_spatial
    M2, sp2 = M.copy(), adata_sp.copy()
    M2.var_names = ["s0", "s0", "s1"]
    sp2.obs_names = ["s1", "s0", "s0"]
    with pytest.raises(ValueError, match="repeats a spot name"):
        dt.pp.wasserstein(M2, sp2, verbose=False, n_jobs=1)


@pytest.mark.parametrize("unnamed", ["mapping", "spatial"])
def test_a_default_index_on_either_side_keeps_the_positional_reading(mapping_and_spatial,
                                                                       unnamed):
    M, adata_sp = mapping_and_spatial
    ref = dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=1).uns["distances"]
    M2, sp2 = M.copy(), adata_sp.copy()
    if unnamed == "mapping":
        M2.var_names = ["0", "1", "2"]
    else:
        sp2.obs_names = ["0", "1", "2"]
    out = dt.pp.wasserstein(M2, sp2, verbose=False, n_jobs=1).uns["distances"]
    np.testing.assert_allclose(out, ref, atol=1e-6)


def test_undefined_pairs_are_nan(mapping_and_spatial, monkeypatch):
    """A cell without spots has no distance (silently); a non-finite EMD is NaN and warned."""
    import ot

    M, adata_sp = mapping_and_spatial
    M = M.copy()
    M.X[2] = 0.0
    dc = dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=1).uns["distances"]
    np.testing.assert_allclose(dc, [3.0, np.nan, np.nan])

    monkeypatch.setattr(ot, "emd2", lambda *a, **k: (np.nan, {"result_code": 1}))
    with pytest.warns(RuntimeWarning, match="1 of 3 cell pairs had a non-finite EMD"):
        dc = dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=1).uns["distances"]
    assert np.isnan(dc).all()


def _spread_mapping(n_cells=6, n_spots=40, power=4, seed=0):
    """A random mapping whose cells spread over many spots, on random coordinates."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, n_spots)) ** power
    X /= X.sum(1, keepdims=True)
    M = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]))
    sp_ = ad.AnnData(X=np.zeros((n_spots, 1)))
    sp_.obsm["spatial"] = rng.random((n_spots, 2)) * 500
    return M, sp_


def test_unconverged_pairs_are_undefined_and_counted(monkeypatch):
    """A pair that hits the iteration cap is NaN and counted: its early-stopped cost bounds nothing."""
    import ot

    M, sp_ = _spread_mapping()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exact = dt.pp.wasserstein(M, sp_, verbose=False, n_jobs=1).uns["distances"]
    emd2 = ot.emd2
    monkeypatch.setattr(ot, "emd2", lambda a, b, M, **kw: emd2(a, b, M, numItermax=1, **kw))
    with pytest.warns(RuntimeWarning, match="15 of 15 cell pairs hit the EMD iteration cap"):
        capped = dt.pp.wasserstein(M, sp_, verbose=False, n_jobs=1).uns["distances"]
    assert np.isfinite(exact).all() and np.isnan(capped).all()


def test_a_nearly_uniform_mapping_is_warned_about(mapping_and_spatial):
    M, sp_ = _spread_mapping(power=1)
    with pytest.warns(UserWarning, match="Mapping nearly uniform"):
        dt.pp.wasserstein(M, sp_, verbose=False, n_jobs=1)

    M, adata_sp = mapping_and_spatial
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=1)
