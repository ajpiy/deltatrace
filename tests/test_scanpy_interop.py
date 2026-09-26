"""Every result a dt.pl function colors by is also a plain obs column, so scanpy's
embedding plots draw the same data from the same object."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _line_toy import line_toy

import deltatrace as dt


@pytest.fixture(scope="module")
def run():
    adata, adata_dist = line_toy(n=240)
    t = np.linspace(0, 1, adata.n_obs)
    i = np.arange(adata.n_obs)
    adata.obs["clusters"] = pd.Categorical(
        np.where(t < 1 / 3, "source", np.where(t < 2 / 3, "mid",
                                                np.where(i % 2 == 0, "A", "B"))))
    g = dt.pp.graphs(adata, adata_dist, alpha=0.9, p_keep=0.3, verbose=0)
    res = dt.tl.trajectory(g, ["source"], ["A", "B"], basis="spatial", p_end=0.2,
                           verbose=0)
    return g, res


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_lineages_writes_one_fate_column_per_lineage(run):
    """obs['fate_<lineage>'] is column k of obsm['path_cellweights'], NaN off-path."""
    g, _ = run
    names = [str(n) for n in g.uns["path_lineages"]]
    W = np.asarray(g.obsm["path_cellweights"], float)
    on = ~g.obs["lineage_confidence"].isna().to_numpy()
    for k, name in enumerate(names):
        col = g.obs[f"fate_{name}"].to_numpy(float)
        np.testing.assert_allclose(col[on], W[on, k])
        assert np.isnan(col[~on]).all()
    # they sum to the same thing as the confidence's argmax column
    stacked = np.column_stack([g.obs[f"fate_{n}"].to_numpy(float) for n in names])
    np.testing.assert_allclose(np.nanmax(stacked, axis=1)[on],
                               g.obs["lineage_confidence"].to_numpy(float)[on])


def test_fate_column_names_are_h5_safe():
    """A cluster called "L5/6" must not become an obs key with a slash (h5 stores are
    dropping those); it is written as fate_L5_6."""
    adata, adata_dist = line_toy(n=150)
    adata.obs["clusters"] = adata.obs["clusters"].astype(str).replace({"sink": "L5/6"})
    g = dt.pp.graphs(adata, adata_dist, alpha=0.9, p_keep=0.3, verbose=0)
    dt.tl.trajectory(g, ["source"], ["L5/6"], basis="spatial", p_end=0.2, verbose=0)
    assert "fate_L5_6" in g.obs and not any("/" in c for c in g.obs.columns)
    assert list(g.uns["path_lineages"]) == ["L5/6"]           # the name itself is kept


def test_lineages_refuses_a_terminal_named_leakage(run):
    g, _ = run
    obj = g.copy()
    obj.obs["clusters"] = obj.obs["clusters"].astype(str).replace({"A": "leakage"})
    with pytest.raises(ValueError, match="fate_leakage"):
        dt.tl.lineages(obj, ["source"], ["leakage", "B"], cluster_key="clusters",
                       verbose=0)


def test_endpoints_writes_a_categorical_endpoint_column(run):
    g, res = run
    col = g.obs["endpoint_intr"]
    assert list(col.cat.categories) == ["source", "intermediate", "sink"]
    src, snk = res["endpoints"]["source_indices"], res["endpoints"]["sink_indices"]
    assert set(np.flatnonzero(col.to_numpy() == "source")) == set(src)
    assert set(np.flatnonzero(col.to_numpy() == "sink")) == set(snk)
    assert (col.to_numpy() == "intermediate").sum() == g.n_obs - len(src) - len(snk)
    assert g.uns["endpoint_intr_colors"] == ["#3B4CC0", "lightgray", "#B40426"]


def test_scanpy_embedding_draws_every_deltatrace_column(run):
    import scanpy as sc

    g, _ = run
    cols = ["pseudotime", "pseudotime_raw", "lineage", "lineage_confidence",
            "fate_leakage", "endpoint_intr", *[f"fate_{n}" for n in g.uns["path_lineages"]]]
    axes = sc.pl.embedding(g, basis="spatial", color=cols, show=False)
    assert len(axes) == len(cols)
    # and the endpoint panel uses the pl.endpoints palette
    ep_ax = axes[cols.index("endpoint_intr")]
    legend = ep_ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["source", "intermediate", "sink"]
