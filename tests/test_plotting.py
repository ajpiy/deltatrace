"""dt.pl — the plotting surface.

These do not check what the pictures look like; they pin the contracts that were broken
and are cheap to regress: the caller's AnnData is not mutated, a supplied ``ax`` is the
one drawn on and saved, and the legend survives source-only mode (no sinks).
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")                        # no display in CI; must precede pyplot
import matplotlib.pyplot as plt
from _line_toy import line_toy

import deltatrace as dt


@pytest.fixture(scope="module")
def run():
    """A finished trajectory on the toy, plus the endpoints dict pl.endpoints needs."""
    ad_g = dt.pp.graphs(*line_toy(n=120), alpha=0.9, p_keep=0.3, verbose=0)
    res = dt.tl.trajectory(ad_g, ["source"], ["sink"], basis="spatial",
                           p_end=0.2, verbose=0)
    return ad_g, res["endpoints"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------------------ pl.pseudotime
def test_pseudotime_returns_fig_and_ax(run):
    ad_g, _ = run
    fig, ax = dt.pl.pseudotime(ad_g, basis="spatial", show=False)
    assert fig is ax.figure


def _record_savefig(monkeypatch):
    """Capture which Figure savefig was invoked on.

    Checking only that the file exists does NOT discriminate: the pre-fix code called
    plt.savefig, which wrote the pyplot *current* figure — a perfectly non-empty but
    BLANK png. The figure identity is the thing worth pinning.
    """
    from matplotlib.figure import Figure

    seen = []
    real = Figure.savefig

    def spy(self, *a, **k):
        seen.append(self)
        return real(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", spy)
    return seen


def test_pseudotime_draws_on_a_supplied_ax(run):
    """The ax= contract: the scatter goes on the supplied axes, not the current figure."""
    ad_g, _ = run
    other = plt.figure()                     # becomes the current figure
    target_fig, target_ax = plt.subplots()
    plt.figure(other.number)                 # make the DECOY current again
    fig, ax = dt.pl.pseudotime(ad_g, basis="spatial", ax=target_ax, show=False)
    assert ax is target_ax and fig is target_fig
    assert len(target_ax.collections) > 0    # the scatter landed here
    assert len(other.axes) == 0              # ...and nothing landed on the decoy


def test_pseudotime_colorbar_does_not_warn_about_a_foreign_figure(run, recwarn):
    """plt.colorbar attaches to ax.figure but warns 'Adding colorbar to a different
    Figure' when another figure is current; fig.colorbar does not. The warning is the
    only observable difference, so it is what the test has to assert on."""
    ad_g, _ = run
    _, target_ax = plt.subplots()
    plt.figure()                             # decoy current figure
    dt.pl.pseudotime(ad_g, basis="spatial", ax=target_ax, show=False)
    assert not [w for w in recwarn if "different Figure" in str(w.message)]


def test_pseudotime_saves_the_supplied_figure(run, tmp_path, monkeypatch):
    ad_g, _ = run
    target_fig, target_ax = plt.subplots()
    plt.figure()                             # decoy current figure
    seen = _record_savefig(monkeypatch)
    out = tmp_path / "pt.png"
    dt.pl.pseudotime(ad_g, basis="spatial", ax=target_ax, save=str(out), show=False)
    assert seen == [target_fig]              # not the decoy
    assert out.exists() and out.stat().st_size > 0


def test_pseudotime_saves_the_root_figure_of_a_subfigure_ax(run, tmp_path, monkeypatch):
    """ax.get_figure() is a SubFigure when the caller used fig.subfigures(); SubFigure has
    no savefig, so saving must walk up to the root Figure instead of raising."""
    ad_g, _ = run
    root = plt.figure()
    sub_ax = root.subfigures(1, 2)[0].subplots()
    seen = _record_savefig(monkeypatch)
    out = tmp_path / "sub.png"
    fig, _ = dt.pl.pseudotime(ad_g, basis="spatial", ax=sub_ax, save=str(out), show=False)
    assert type(fig).__name__ == "SubFigure"   # what the caller gets back
    assert seen == [root]                      # what was actually written
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("arrows", ["quiver", "stream", "none"])
def test_pseudotime_arrow_styles(run, arrows):
    ad_g, _ = run
    fig, _ = dt.pl.pseudotime(ad_g, basis="spatial", arrows=arrows, show=False)
    assert fig is not None


def test_pseudotime_rejects_an_unknown_arrow_style(run):
    ad_g, _ = run
    with pytest.raises(ValueError, match="arrows must be"):
        dt.pl.pseudotime(ad_g, basis="spatial", arrows="wiggly", show=False)


# -------------------------------------------------------------------------- pl.endpoints
def test_endpoints_does_not_mutate_the_callers_object(run):
    """The scratch category and scanpy's derived color map must not survive the call."""
    ad_g, graph = run
    before_obs, before_uns = set(ad_g.obs.columns), set(ad_g.uns)
    dt.pl.endpoints(graph, ad_g, basis="spatial", graph="intr", show=False)
    assert set(ad_g.obs.columns) == before_obs
    assert set(ad_g.uns) == before_uns
    assert "_endpoint_class" not in ad_g.obs
    assert "_endpoint_class_colors" not in ad_g.uns


def test_endpoints_never_calls_scanpy(run, monkeypatch):
    """pl.endpoints used to route its background through sc.pl.embedding via scratch
    obs/obsm/uns keys; it now draws everything itself, like the other pl functions."""
    import scanpy as sc

    ad_g, graph = run

    def boom(*a, **k):
        raise AssertionError("pl.endpoints must not call sc.pl.embedding")

    monkeypatch.setattr(sc.pl, "embedding", boom)
    before = (set(ad_g.obs.columns), set(ad_g.obsm), set(ad_g.uns))
    _, ax = dt.pl.endpoints(graph, ad_g, basis="spatial", graph="intr", show=False)
    assert (set(ad_g.obs.columns), set(ad_g.obsm), set(ad_g.uns)) == before
    from matplotlib.collections import PathCollection

    n_pts = sum(len(c.get_offsets()) for c in ax.collections
                if isinstance(c, PathCollection))
    assert n_pts == ad_g.n_obs + len(graph["source_indices"]) + len(graph["sink_indices"])


def test_endpoints_background_uses_class_colors(run):
    """Source cells are drawn in source_color even in the small background layer, so
    the picture reads without the overlay."""
    from matplotlib.colors import to_rgba

    ad_g, graph = run
    _, ax = dt.pl.endpoints(graph, ad_g, basis="spatial", graph="intr", show=False,
                            source_color="#112233", show_edges=False)
    src = set(graph["source_indices"])
    src_pts = ad_g.obsm["spatial"][sorted(src)]
    hits = [c for c in ax.collections
            if np.allclose(c.get_facecolor()[0], to_rgba("#112233"))
            and len(c.get_offsets()) == len(src)]
    assert hits, "no collection carries the source cells in source_color"
    np.testing.assert_allclose(np.sort(np.asarray(hits[0].get_offsets()), axis=0),
                               np.sort(src_pts, axis=0))


def test_endpoints_legend_survives_source_only(run):
    """With sink_clusters=None there is no 'sink' handle; the old fixed [1, 0, 2]
    permutation raised IndexError on the two that exist."""
    ad_g, _ = run
    obj = ad_g.copy()
    graph = dt.tl.endpoints(obj, obj, ["source"], None, cluster_key="clusters",
                            graph="intr", p_end=0.2, verbose=0)
    assert graph["sink_indices"] == []
    _, ax = dt.pl.endpoints(graph, obj, basis="spatial", graph="intr", show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "sink" not in labels
    assert "source" in labels


def test_endpoints_legend_is_ordered_source_intermediate_sink(run):
    ad_g, graph = run
    _, ax = dt.pl.endpoints(graph, ad_g, basis="spatial", graph="intr", show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["source", "intermediate", "sink"]


# ------------------------------------------------------------------------ shared invariant
def test_no_plot_function_leaves_stray_axes(run):
    """Each call creates exactly one figure when no ax is supplied."""
    ad_g, graph = run
    plt.close("all")
    dt.pl.pseudotime(ad_g, basis="spatial", show=False)
    assert len(plt.get_fignums()) == 1
    plt.close("all")
    dt.pl.endpoints(graph, ad_g, basis="spatial", graph="intr", show=False)
    assert len(plt.get_fignums()) == 1


def test_pseudotime_pseudotime_values_are_what_is_colored(run):
    """Guards against the plot reading a stale/flat key: the scatter's array must be
    obs['pseudotime'] itself."""
    ad_g, _ = run
    _, ax = dt.pl.pseudotime(ad_g, basis="spatial", arrows="none", show=False)
    scat = ax.collections[0]
    np.testing.assert_allclose(
        np.asarray(scat.get_array(), float),
        ad_g.obs["pseudotime"].to_numpy(float),
    )


def test_pseudotime_nan_cells_are_drawn_in_na_color_underneath(run):
    """A NaN pseudotime used to hit matplotlib's transparent 'bad' color and vanish."""
    from matplotlib.colors import to_rgba

    ad_g, _ = run
    obj = ad_g.copy()
    obj.obs.loc[obj.obs_names[:7], "pseudotime"] = np.nan
    _, ax = dt.pl.pseudotime(obj, basis="spatial", arrows="none", show=False,
                             na_color="#0a0b0c")
    na = [c for c in ax.collections if len(c.get_offsets()) == 7]
    assert len(na) == 1
    assert np.allclose(na[0].get_facecolor()[0], to_rgba("#0a0b0c"))
    finite = [c for c in ax.collections if len(c.get_offsets()) == obj.n_obs - 7]
    assert len(finite) == 1 and na[0].get_zorder() < finite[0].get_zorder()
    assert not np.isnan(np.asarray(finite[0].get_array(), float)).any()


def test_pseudotime_colorbar_and_limits_are_controllable(run):
    ad_g, _ = run
    fig, _ = dt.pl.pseudotime(ad_g, basis="spatial", arrows="none", show=False,
                              colorbar_label="pseudotime")
    assert len(fig.axes) == 2                                   # axes + colorbar
    assert fig.axes[1].get_ylabel() == "pseudotime"
    fig, _ = dt.pl.pseudotime(ad_g, basis="spatial", arrows="none", show=False)
    assert fig.axes[1].get_ylabel() == ""                       # unlabelled by default
    fig2, ax2 = dt.pl.pseudotime(ad_g, basis="spatial", arrows="none", show=False,
                                 colorbar=False, vmin=0.2, vmax=0.4, title="t")
    assert len(fig2.axes) == 1
    assert ax2.collections[0].get_clim() == (0.2, 0.4)
    assert ax2.get_title() == "t"


def test_pseudotime_default_marker_size_follows_scanpy(run):
    """No node_size -> scanpy's 120000 / n_obs, not matplotlib's 36 pt^2."""
    ad_g, _ = run
    _, ax = dt.pl.pseudotime(ad_g, basis="spatial", arrows="none", show=False)
    assert ax.collections[0].get_sizes()[0] == pytest.approx(120000 / ad_g.n_obs)
    _, ax = dt.pl.pseudotime(ad_g, basis="spatial", arrows="none", show=False,
                             node_size=3)
    assert ax.collections[0].get_sizes()[0] == 3


# ---------------------------------------------------------------------------- pl.velocity
def test_velocity_draws_the_same_arrows_pseudotime_does(run):
    """pl.pseudotime's arrow layer IS pl.velocity: same quiver offsets and vectors."""
    ad_g, _ = run
    _, ax_pt = dt.pl.pseudotime(ad_g, basis="spatial", arrows="quiver", arrow_scale=2.0,
                                show=False)
    _, ax = plt.subplots()
    ax.scatter(*ad_g.obsm["spatial"].T)
    _, ax_v = dt.pl.velocity(ad_g, ax=ax, kind="quiver", arrow_scale=2.0)
    assert ax_v is ax
    q_pt = next(c for c in ax_pt.collections if hasattr(c, "U"))
    q_v = next(c for c in ax.collections if hasattr(c, "U"))
    np.testing.assert_allclose(q_v.get_offsets(), q_pt.get_offsets())
    np.testing.assert_allclose(q_v.U, q_pt.U)
    np.testing.assert_allclose(q_v.V, q_pt.V)


def test_velocity_on_a_scanpy_axes(run):
    """The interoperability story: scanpy draws the cells, DeltaTrace adds the arrows."""
    import scanpy as sc

    ad_g, _ = run
    ax = sc.pl.embedding(ad_g, basis="spatial", color="pseudotime", show=False)
    n_before = len(ax.collections)
    _, ax_v = dt.pl.velocity(ad_g, ax=ax, kind="quiver")
    assert ax_v is ax and len(ax.collections) == n_before + 1


@pytest.mark.parametrize("kind", ["quiver", "stream", "raw"])
def test_velocity_kinds_and_own_figure(run, kind):
    ad_g, _ = run
    plt.close("all")
    fig, ax = dt.pl.velocity(ad_g, kind=kind, raw_subsample=0.5, raw_seed=0)
    assert len(plt.get_fignums()) == 1 and ax.figure is fig


def test_velocity_warns_when_scanpy_would_pick_the_other_obsm_key(run):
    """obsm has both 'X_spatial' and 'spatial' with different values: DeltaTrace used
    X_spatial for the grid, scanpy's basis='spatial' would scatter on 'spatial'."""
    ad_g, _ = run
    obj = ad_g.copy()
    obj.obsm["X_spatial"] = np.asarray(obj.obsm["spatial"]) * 3.0   # DeltaTrace's pick
    dt.tl.compute_velocity_grid(obj, basis="spatial", grid_size=(8, 8), k=5, verbose=0)
    _, ax = plt.subplots()
    with pytest.warns(UserWarning, match="Pass basis='X_spatial' to scanpy"):
        dt.pl.velocity(obj, ax=ax)
    import warnings

    # naming the X_ key says "I drew on the grid's coordinates": no warning
    _, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dt.pl.velocity(obj, basis="X_spatial", ax=ax)
    # identical arrays under both keys: nothing to warn about
    obj.obsm["X_spatial"] = np.asarray(obj.obsm["spatial"]).copy()
    _, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dt.pl.velocity(obj, ax=ax)


def test_velocity_refuses_a_basis_the_grid_was_not_computed_on(run):
    ad_g, _ = run
    with pytest.raises(ValueError, match="computed on basis='spatial'"):
        dt.pl.velocity(ad_g, basis="umap")
    with pytest.raises(ValueError, match="kind must be"):
        dt.pl.velocity(ad_g, kind="arrows")


# ---------------------------------------------------------------------------- pl.lineages
@pytest.fixture(scope="module")
def multi_lineage():
    """Toy with THREE terminals, so lineage coloring has something to distinguish."""
    import pandas as pd

    adata, adata_dist = line_toy(n=240)
    t = np.linspace(0, 1, adata.n_obs)
    i = np.arange(adata.n_obs)
    adata.obs["clusters"] = pd.Categorical(
        np.where(t < 1 / 3, "source",
                 np.where(t < 2 / 3, "mid",
                          np.where(i % 3 == 0, "A", np.where(i % 3 == 1, "B", "C"))))
    )
    g = dt.pp.graphs(adata, adata_dist, alpha=0.9, p_keep=0.3, verbose=0)
    dt.tl.trajectory(g, ["source"], ["A", "B", "C"], basis="spatial",
                     p_end=0.2, verbose=0)
    return g


def test_lineages_uses_the_requested_color_per_lineage(multi_lineage):
    """The whole point of lineage_colors: the color asked for is the color drawn."""
    import matplotlib.colors as mcolors

    want = {"A": "#bb5566", "B": "#ddaa33", "C": "#004488"}
    _, ax = dt.pl.lineages(multi_lineage, basis="spatial", lineage_colors=want,
                           show=False)
    drawn = [tuple(np.round(c.get_facecolor()[0][:3], 3))
             for c in ax.collections if len(c.get_facecolor())]
    for hexcol in want.values():
        assert any(np.allclose(d, mcolors.to_rgb(hexcol), atol=1e-3) for d in drawn), \
            f"{hexcol} was requested but never drawn"


def test_lineages_default_to_the_colorblind_safe_okabe_ito_palette(multi_lineage):
    """Without lineage_colors, the three lineages are drawn in the first three Okabe-Ito
    colors, in name order."""
    import matplotlib.colors as mcolors

    _, ax = dt.pl.lineages(multi_lineage, basis="spatial", show=False)
    drawn = [tuple(np.round(c.get_facecolor()[0][:3], 3))
             for c in ax.collections if len(c.get_facecolor())]
    for hexcol in ("#0072B2", "#E69F00", "#009E73"):
        assert any(np.allclose(d, mcolors.to_rgb(hexcol), atol=1e-3) for d in drawn), \
            f"{hexcol} (Okabe-Ito) was never drawn"


def test_lineages_rejects_an_unknown_lineage_name(multi_lineage):
    """A typo must raise, not silently produce a figure in fallback colors."""
    with pytest.raises(ValueError, match="not lineages of this run"):
        dt.pl.lineages(multi_lineage, basis="spatial",
                       lineage_colors={"nosuchlineage": "#000000"}, show=False)


def test_lineages_partial_override_falls_back_to_cmap(multi_lineage):
    _, ax = dt.pl.lineages(multi_lineage, basis="spatial",
                           lineage_colors={"A": "#bb5566"}, show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"A", "B", "C"} <= set(labels)


def test_lineages_legend_lists_every_drawn_lineage(multi_lineage):
    _, ax = dt.pl.lineages(multi_lineage, basis="spatial", show=False)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    for name in multi_lineage.uns["path_lineages"]:
        assert str(name) in labels


def test_lineages_confidence_alpha_is_per_cell(multi_lineage):
    """With confidence_alpha the opacity varies per cell; without it, it is constant."""
    _, ax_on = dt.pl.lineages(multi_lineage, basis="spatial", show=False)
    _, ax_off = dt.pl.lineages(multi_lineage, basis="spatial",
                               confidence_alpha=False, show=False)
    a_on = [c.get_alpha() for c in ax_on.collections]
    assert any(np.ndim(a) > 0 and np.unique(np.round(a, 6)).size > 1 for a in a_on)
    for c in ax_off.collections:
        a = c.get_alpha()
        assert np.ndim(a) == 0 or np.unique(np.round(a, 6)).size == 1


def test_lineages_falls_back_to_the_dt_grid_basis(multi_lineage):
    fig, ax = dt.pl.lineages(multi_lineage, show=False)
    assert fig is ax.figure


def test_lineages_requires_tl_lineages_to_have_run():
    bare, _ = line_toy(n=40)
    with pytest.raises(KeyError, match=r"run deltatrace\.tl\.lineages"):
        dt.pl.lineages(bare, basis="spatial", show=False)


def test_lineages_saves_the_supplied_figure(multi_lineage, tmp_path, monkeypatch):
    target_fig, target_ax = plt.subplots()
    plt.figure()                                      # decoy
    seen = _record_savefig(monkeypatch)
    out = tmp_path / "lin.png"
    dt.pl.lineages(multi_lineage, basis="spatial", ax=target_ax, save=str(out),
                   show=False)
    assert seen == [target_fig]
    assert out.exists() and out.stat().st_size > 0


# ------------------------------------------------------------------ coord_scale coherence
def _quiver(ax):
    from matplotlib.quiver import Quiver

    qs = [c for c in ax.collections if isinstance(c, Quiver)]
    assert len(qs) == 1, f"expected exactly one quiver, got {len(qs)}"
    return qs[0]


def test_pseudotime_coord_scale_moves_the_grid_arrows_with_the_cells(run):
    """The grid lives in the embedding's units, so coord_scale must reach it too.

    The cell scatter was scaled but uns['dt_grid']['Xg'/'Yg'] was read raw, so at any
    coord_scale != 1 the arrows sat in a different coordinate system from the cells; and
    with angles/scale_units='xy', scale=1 their LENGTHS were short by the same factor.
    """
    ad_g, _ = run
    s = 3.0
    _, ax1 = dt.pl.pseudotime(ad_g, basis="spatial", coord_scale=1.0, show=False)
    _, axs = dt.pl.pseudotime(ad_g, basis="spatial", coord_scale=s, show=False)

    cells1 = np.asarray(ax1.collections[0].get_offsets(), float)
    cells_s = np.asarray(axs.collections[0].get_offsets(), float)
    np.testing.assert_allclose(cells_s, cells1 * s)          # the cells scale (they did)

    q1, qs = _quiver(ax1), _quiver(axs)
    assert np.asarray(q1.get_offsets()).size > 0             # arrows were actually drawn
    np.testing.assert_allclose(np.asarray(qs.get_offsets(), float),
                               np.asarray(q1.get_offsets(), float) * s)
    np.testing.assert_allclose(np.asarray(qs.U, float), np.asarray(q1.U, float) * s)
    np.testing.assert_allclose(np.asarray(qs.V, float), np.asarray(q1.V, float) * s)


def test_pseudotime_coord_scale_moves_the_raw_arrows_with_the_cells(run):
    ad_g, _ = run
    s = 3.0
    _, ax1 = dt.pl.pseudotime(ad_g, basis="spatial", arrows="raw", coord_scale=1.0,
                              show=False)
    _, axs = dt.pl.pseudotime(ad_g, basis="spatial", arrows="raw", coord_scale=s,
                              show=False)
    q1, qs = _quiver(ax1), _quiver(axs)
    np.testing.assert_allclose(np.asarray(qs.U, float), np.asarray(q1.U, float) * s)


def test_pseudotime_coord_scale_default_is_inert(run):
    """At the default the drawn arrows must be the STORED grid, untouched.

    Comparing two default-argument calls would be tautological -- it would still pass if
    coord_scale were applied twice -- so this pins the drawn values against
    uns["dt_grid"] and the raw embedding instead.
    """
    ad_g, _ = run
    _, ax = dt.pl.pseudotime(ad_g, basis="spatial", arrow_scale=1.0, show=False)
    grid = ad_g.uns["dt_grid"]
    m = grid["mask"]
    q = _quiver(ax)
    exp_xy = np.column_stack([grid["Xg"][m], grid["Yg"][m]])
    np.testing.assert_array_equal(np.asarray(q.get_offsets(), float), exp_xy)
    np.testing.assert_array_equal(np.asarray(q.U, float), grid["U"][m])
    np.testing.assert_array_equal(np.asarray(q.V, float), grid["V"][m])
    # ...and the cell scatter is the raw embedding
    cells = np.asarray(ax.collections[0].get_offsets(), float)
    np.testing.assert_array_equal(cells, np.asarray(ad_g.obsm["spatial"], float))


def test_endpoints_coord_scale_reaches_the_background_cells(run):
    """Every layer -- background, endpoints, edges -- is scaled by the same factor (the
    background once went through sc.pl.embedding and stayed at 1x)."""
    ad_g, graph = run
    s = 4.0
    _, ax1 = dt.pl.endpoints(graph, ad_g, basis="spatial", graph="intr", coord_scale=1.0,
                             show=False)
    _, axs = dt.pl.endpoints(graph, ad_g, basis="spatial", graph="intr", coord_scale=s,
                             show=False)

    def _all_points(ax):
        return np.vstack([np.asarray(c.get_offsets(), float) for c in ax.collections])

    np.testing.assert_allclose(_all_points(axs), _all_points(ax1) * s)
    assert "X__dt_endpoint_scaled" not in ad_g.obsm      # no scratch basis, ever


# ------------------------------------------------- pl.endpoints spring-layout fallback
def test_endpoints_falls_back_to_a_spring_layout_when_the_basis_is_missing(run):
    """The documented fallback: a missing basis lays the graph out with networkx (it was
    once computed and then thrown away by a scanpy call on the missing basis)."""
    import networkx as nx

    ad_g, graph = run
    with pytest.warns(UserWarning, match="spring layout") as rec:
        fig, ax = dt.pl.endpoints(graph, ad_g, basis="no_such_basis", graph="intr", show=False)
    assert rec[0].filename == __file__                  # said, and pointed at the caller
    assert fig is not None
    pts = np.vstack([np.asarray(c.get_offsets(), float) for c in ax.collections])
    assert len(pts) >= ad_g.n_obs                       # every cell was drawn
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["source", "intermediate", "sink"]

    # The layout must actually be USED, not merely computed and thrown away -- that was
    # the bug. Degenerate coords (all cells stacked on one point) would satisfy a mere
    # count assertion, so pin the coordinates against the layout itself.
    assert pts.std(axis=0).min() > 0, "fallback coords are degenerate"
    assert len({tuple(p) for p in pts}) >= ad_g.n_obs, "cells share positions"
    pos = nx.spring_layout(graph["G"].to_networkx().to_undirected(), seed=42)
    expect = np.array([pos[i] for i in range(ad_g.n_obs)])
    drawn = {tuple(np.round(p, 12)) for p in pts}
    assert {tuple(np.round(p, 12)) for p in expect} <= drawn


def test_endpoints_fallback_honors_coord_scale(run):
    ad_g, graph = run
    s = 5.0
    with pytest.warns(UserWarning, match="spring layout"):
        _, ax1 = dt.pl.endpoints(graph, ad_g, basis="no_such_basis", graph="intr",
                                 coord_scale=1.0, show=False)
        _, axs = dt.pl.endpoints(graph, ad_g, basis="no_such_basis", graph="intr",
                                 coord_scale=s, show=False)
    p1 = np.vstack([np.asarray(c.get_offsets(), float) for c in ax1.collections])
    ps = np.vstack([np.asarray(c.get_offsets(), float) for c in axs.collections])
    assert p1.std(axis=0).min() > 0, "degenerate coords make the scaling check vacuous"
    np.testing.assert_allclose(ps, p1 * s)



def test_endpoints_basis_none_finds_the_tissue(run):
    """basis=None resolves like the other pl functions (here: obsm['spatial']), silently."""
    import warnings
    ad_g, graph = run
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, ax = dt.pl.endpoints(graph, ad_g, graph="intr", show=False)
    pts = np.vstack([np.asarray(c.get_offsets(), float) for c in ax.collections])
    xy = np.asarray(ad_g.obsm["spatial"], float)
    assert {tuple(np.round(p, 9)) for p in xy} <= {tuple(np.round(p, 9)) for p in pts}



def test_endpoints_titles_the_graph_it_was_sampled_on_even_with_two_stored(run):
    """pl.endpoints reads the graph from the endpoints result, so an object holding both
    'expr' and 'intr' needs no graph= just to title the plot."""
    ad_g, graph = run
    _, ax = dt.pl.endpoints(graph, ad_g, basis="spatial", show=False)
    assert ax.get_title() == "Endpoints: intr"
