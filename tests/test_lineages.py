"""Pin deltatrace.tl.lineages (absorbing-chain fates) on a hand-computable toy.

Toy flow DAG:  S(0) -> M(1) -> {A(2), B(3)}.  M is a bipotent split, so the chain
must report M (and the source S it feeds) as genuinely split by the branch masses,
not 100 % one fate. Fates are computed by one backward pass and are unique.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from _line_toy import line_toy

import deltatrace as dt


def _toy(mA=0.5, mB=0.5):
    obs = pd.DataFrame(
        {"clusters": ["src", "mid", "termA", "termB"]},
        index=["c0", "c1", "c2", "c3"],
    )
    a = ad.AnnData(X=np.zeros((4, 2), np.float32), obs=obs)
    a.obs["pseudotime"] = [0.0, 0.5, 1.0, 1.0]
    a.uns["dt_flow"] = {
        "edges": np.array([[0, 1], [1, 2], [1, 3]], np.int64),
        "J": np.array([mA + mB, mA, mB], float),
    }
    return a


def test_chain_bipotent_split():
    a = _toy(0.5, 0.5)
    dt.tl.lineages(a, ["src"], ["termA", "termB"], cluster_key="clusters", verbose=False)

    assert list(a.uns["path_lineages"]) == ["termA", "termB"]
    W = a.obsm["path_cellweights"]
    assert W.shape == (4, 2)
    # M and the source S it feeds are genuinely bipotent (0.5/0.5); A/B are pure.
    np.testing.assert_allclose(
        W, [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], atol=1e-9
    )
    on = W.sum(1) > 1e-12
    np.testing.assert_allclose(W[on].sum(1), 1.0)          # fate rows sum to 1 on-path
    np.testing.assert_allclose(a.obs["fate_leakage"].to_numpy(), 0.0, atol=1e-9)
    assert list(a.obs["lineage"]) == ["termA", "termA", "termA", "termB"]  # ties -> first


def test_chain_asymmetric_fate():
    a = _toy(0.7, 0.3)
    dt.tl.lineages(a, ["src"], ["termA", "termB"], cluster_key="clusters", verbose=False)
    W = a.obsm["path_cellweights"]
    np.testing.assert_allclose(W[1], [0.7, 0.3], atol=1e-9)   # bipotent cell M
    np.testing.assert_allclose(W[0], [0.7, 0.3], atol=1e-9)   # source inherits M's mix
    assert a.obs["lineage_confidence"].to_numpy()[1] == pytest.approx(0.7)


def test_chain_no_source_raises():
    a = _toy()
    with pytest.raises(ValueError, match="No source cells"):
        dt.tl.lineages(a, ["nope"], ["termA", "termB"], cluster_key="clusters", verbose=False)


def test_source_only_requires_omega_term():
    """terminal_clusters=None without omega_term errors and suggests a value."""
    a = _toy()
    with pytest.raises(ValueError, match="omega_term"):
        dt.tl.lineages(a, ["src"], None, cluster_key="clusters", verbose=False)


def test_widest_path_toy():
    a = _toy(0.7, 0.3)
    # weightiest route S -> A follows the heavier branch and exists
    path, cap = dt.tl.widest_path(a, 0, 2)
    assert path == [0, 1, 2]
    assert cap == pytest.approx(0.7)


# ------------------------------------------------- the missing-data convention (3 columns)
def _toy_with_offpath_cells():
    """Extend the toy with a cell the flow never reaches.

    c4 is in no path of the DAG, so the chain gives it b == 0: no fate, no confidence and
    no leakage. It stands in for the 13 cells the stock line toy (``tests/_line_toy.py``) leaves off-path.
    """
    obs = pd.DataFrame(
        {"clusters": ["src", "mid", "termA", "termB", "orphan"]},
        index=["c0", "c1", "c2", "c3", "c4"],
    )
    a = ad.AnnData(X=np.zeros((5, 2), np.float32), obs=obs)
    a.obs["pseudotime"] = [0.0, 0.5, 1.0, 1.0, 0.5]
    a.uns["dt_flow"] = {
        "edges": np.array([[0, 1], [1, 2], [1, 3]], np.int64),
        "J": np.array([1.0, 0.5, 0.5], float),
    }
    return a


def test_fate_leakage_and_lineage_confidence_agree_on_what_is_missing():
    """One missing-data convention across all three fate columns.

    fate_leakage was written as ``1 - b`` unconditionally, so a cell the flow never
    reaches (b == 0) was recorded as exactly 1.0 — "maximally leaky" — while the very same
    cell was "unassigned" in obs['lineage'] and NaN in obs['lineage_confidence'].
    """
    a = _toy_with_offpath_cells()
    dt.tl.lineages(a, ["src"], ["termA", "termB"], cluster_key="clusters", verbose=False)

    leak = a.obs["fate_leakage"].to_numpy(float)
    conf = a.obs["lineage_confidence"].to_numpy(float)
    unassigned = a.obs["lineage"].to_numpy(str) == "unassigned"

    assert unassigned.any() and not unassigned.all()       # the test is not vacuous
    np.testing.assert_array_equal(np.isnan(leak), np.isnan(conf))
    np.testing.assert_array_equal(np.isnan(leak), unassigned)
    assert leak[4] != 1.0 and np.isnan(leak[4])            # the orphan, specifically
    np.testing.assert_allclose(leak[:4], 0.0, atol=1e-9)   # on-path cells keep their value


def test_offpath_cells_do_not_inflate_the_mean_leakage():
    """np.nanmean over the column is the on-path mean — what the verbose=2 line prints.

    With the off-path cells stored as 1.0 they had no NaN to skip, so the reported mean
    was dominated by cells that have no leakage to report.
    """
    a = _toy_with_offpath_cells()
    dt.tl.lineages(a, ["src"], ["termA", "termB"], cluster_key="clusters", verbose=False)
    leak = a.obs["fate_leakage"].to_numpy(float)
    assert np.nanmean(leak) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(leak).sum() == 1                       # and one cell simply has none


def test_fate_leakage_matches_lineage_confidence_on_the_stock_toy():
    """The same contract end to end, on the stock toy (at the pre-0.1.0
    defaults, spelled out so the known off-path count below stays meaningful)."""
    ad_g = dt.pp.graphs(*line_toy(n=150), alpha=0.5, p_keep=0.1, verbose=0)
    dt.tl.trajectory(ad_g, ["source"], ["sink"], basis="spatial", verbose=0, p_end=0.01,
                     p_basis="lcc", sampling_mode="directed")

    leak = ad_g.obs["fate_leakage"].to_numpy(float)
    conf = ad_g.obs["lineage_confidence"].to_numpy(float)
    assert np.isnan(conf).sum() == 13                      # the known off-path cells
    np.testing.assert_array_equal(np.isnan(leak), np.isnan(conf))
    assert np.nanmean(leak) < 1e-3                         # on-path mean, not 0.0867
