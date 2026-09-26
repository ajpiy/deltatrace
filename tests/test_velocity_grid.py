"""tl.compute_velocity_grid: the knobs must reach BOTH arrow products.

The function writes two things — the smoothed grid field (``uns['dt_grid']``) and the
unsmoothed per-cell arrows (``uns['dt_raw_arrows']``). ``transform_scale`` is the
magnitude-compression scale for both, and used to be honored only by the first.
"""

import numpy as np
import pytest
from _line_toy import line_toy

import deltatrace as dt


@pytest.fixture(scope="module")
def solved():
    """A finished flow with no velocity grid yet, so each test computes its own."""
    ad_g = dt.pp.graphs(*line_toy(n=120), alpha=0.9, p_keep=0.3, verbose=0)
    dt.tl.trajectory(ad_g, ["source"], ["sink"], basis="spatial", p_end=0.2,
                     compute_velocity=False, verbose=0)
    return ad_g


def _raw(obj):
    r = obj.uns["dt_raw_arrows"]
    return np.hypot(r["U"], r["V"])[r["mask"]]


def test_transform_scale_reaches_the_raw_arrows(solved):
    """The raw path always recomputed its own 80th percentile, so the user's
    transform_scale was silently dropped — and by then the local variable had been
    overwritten with the grid's default anyway.
    """
    obj = solved.copy()

    dt.tl.compute_velocity_grid(obj, basis="spatial", transform_scale=1e-3, verbose=0)
    small = _raw(obj)
    dt.tl.compute_velocity_grid(obj, basis="spatial", transform_scale=1e2, verbose=0)
    big = _raw(obj)

    assert small.size > 0
    assert not np.allclose(small, big)
    # log1p(mag/ts) shrinks as ts grows, so a larger scale must give shorter arrows
    assert big.sum() < small.sum()


def test_transform_scale_none_still_uses_the_raw_percentile(solved):
    """Inert at the default: None must keep the per-product 80th-percentile behavior."""
    obj = solved.copy()
    dt.tl.compute_velocity_grid(obj, basis="spatial", verbose=0)
    default = _raw(obj)
    dt.tl.compute_velocity_grid(obj, basis="spatial", transform_scale=None, verbose=0)
    np.testing.assert_array_equal(_raw(obj), default)

    # ...and that default is NOT what any particular explicit scale gives, i.e. the
    # explicit branch really is a different code path.
    dt.tl.compute_velocity_grid(obj, basis="spatial", transform_scale=1e-3, verbose=0)
    assert not np.allclose(_raw(obj), default)


def test_transform_scale_does_not_change_the_raw_directions(solved):
    """Only lengths are compressed; the flow directions must be untouched."""
    obj = solved.copy()
    dt.tl.compute_velocity_grid(obj, basis="spatial", transform_scale=1e-3, verbose=0)
    a = obj.uns["dt_raw_arrows"]
    ua, va, m = a["U"].copy(), a["V"].copy(), a["mask"].copy()
    dt.tl.compute_velocity_grid(obj, basis="spatial", transform_scale=1e2, verbose=0)
    b = obj.uns["dt_raw_arrows"]

    ang_a = np.arctan2(va[m], ua[m])
    ang_b = np.arctan2(b["V"][m], b["U"][m])
    np.testing.assert_allclose(ang_a, ang_b, atol=1e-9)
