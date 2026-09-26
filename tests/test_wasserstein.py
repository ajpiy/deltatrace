"""Pin deltatrace.pp.wasserstein to known input -> output."""

import numpy as np
import pytest

import deltatrace as dt


@pytest.mark.parametrize("n_jobs", [1, -1], ids=["serial", "parallel"])
def test_wasserstein_known_distances(mapping_and_spatial, n_jobs):
    M, adata_sp = mapping_and_spatial

    ad_wd = dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=n_jobs)

    # Shape and index aligned to the cells of M.
    assert ad_wd.shape == (3, 3)
    assert list(ad_wd.obs_names) == ["c0", "c1", "c2"]

    WD = ad_wd.X.toarray()
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
    WD = ad_wd.X.toarray()

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


def test_serial_and_parallel_agree_to_float32_precision():
    """n_jobs only sets speed: the parallel path holds the weights as float32, so the two
    agree to float32 precision (relative ~1e-7), on a mapping with real spread."""
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
    a = dt.pp.wasserstein(M, sp_, verbose=False, n_jobs=1).X.toarray()
    b = dt.pp.wasserstein(M, sp_, verbose=False, n_jobs=-1).X.toarray()
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=0)


def test_spots_are_matched_by_name_not_order(mapping_and_spatial):
    """A reordered spatial object (same spots, other order) gives the same distances: the
    mapping's spot columns are matched to the spots by name."""
    M, adata_sp = mapping_and_spatial
    ref = dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=1).X.toarray()
    shuffled = adata_sp[["s2", "s0", "s1"]].copy()
    out = dt.pp.wasserstein(M, shuffled, verbose=False, n_jobs=1).X.toarray()
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
    ref = dt.pp.wasserstein(M, adata_sp, verbose=False, n_jobs=1).X.toarray()
    M2, sp2 = M.copy(), adata_sp.copy()
    if unnamed == "mapping":
        M2.var_names = ["0", "1", "2"]
    else:
        sp2.obs_names = ["0", "1", "2"]
    out = dt.pp.wasserstein(M2, sp2, verbose=False, n_jobs=1).X.toarray()
    np.testing.assert_allclose(out, ref, atol=1e-6)
