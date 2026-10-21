"""Plotting for DeltaTrace results, surfaced through :mod:`deltatrace.pl`.

These functions read results written by :mod:`deltatrace.tl` (pseudotime, velocity
grids, endpoint samples, lineages, the cluster trajectory) and draw with matplotlib
alone; they never import ``scanpy``, ``tangram`` or ``commot``. Everything they color by
is also a plain ``obs`` column, so ``scanpy.pl.embedding`` works on the same object, and
:func:`velocity` adds the arrows to any axes.
"""

from __future__ import annotations

import itertools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ._utils import _PKG_DIR, CellGraph, _resolve_basis, _resolve_graph, _suffix

__all__ = ["pseudotime", "velocity", "endpoints", "lineages",
           "cluster_trajectory"]


class _PlotResult(tuple):
    """A (fig, ax) tuple that forwards attribute access to `ax` for Scanpy compatibility.

    Supports unpacking: `fig, ax = dt.pl.pseudotime(...)`.
    Also allows calling axis methods directly: `dt.pl.pseudotime(...).set_title(...)`.
    """

    def __new__(cls, fig, ax):
        return super().__new__(cls, (fig, ax))

    def __getnewargs__(self):
        return tuple(self)

    @property
    def fig(self):
        return self[0]

    @property
    def ax(self):
        return self[1]

    def __getattr__(self, name):
        return getattr(self[1], name)


def _root_fig(ax):
    """Walk up to the top-level figure that owns ``ax``, since a ``SubFigure`` (from
    ``fig.subfigures(...)``) has no ``savefig``/``tight_layout`` of its own."""
    fig = ax.get_figure()
    while not hasattr(fig, "savefig"):        # SubFigure -> ... -> Figure
        fig = fig.figure
    return fig


def _require_uns(adata, key: str, key_added: str | None, produced_by: str):
    """Fetch ``adata.uns[key]``, or raise a ``ValueError`` naming the ``deltatrace.tl``
    call that would have written it, instead of a bare ``KeyError``."""
    value = adata.uns.get(key)
    if value is None:
        raise ValueError(
            f"No data in adata.uns['{key}']: run deltatrace.tl.{produced_by}()"
            + (f" with key_added={key_added!r}" if key_added else "")
            + " first."
        )
    return value


# ---------------------------------------------------------------------------
# shared drawing helpers
# ---------------------------------------------------------------------------
def _auto_size(n_obs: int) -> float:
    """Marker area scanpy uses when none is given (``120000 / n_obs``), so a plot of
    thousands of cells is readable without passing ``node_size`` every time."""
    return 120000.0 / max(int(n_obs), 1)


def _draw_edges(ax, coords, edges, alpha, width, color="gray", zorder=1):
    """One gray segment per ``(u, v)`` row of ``edges`` between ``coords[u]`` and
    ``coords[v]``; nothing is drawn for an empty edge list."""
    from matplotlib.collections import LineCollection

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if edges.shape[0] == 0:
        return None
    segs = np.stack((coords[edges[:, 0]], coords[edges[:, 1]]), axis=1)
    lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=width, zorder=zorder)
    ax.add_collection(lc)
    return lc


def _warn_if_scanpy_resolves_basis_differently(adata, basis):
    """DeltaTrace reads ``obsm["X_<basis>"]`` before ``obsm["<basis>"]``; scanpy the other
    way round. When both keys exist and differ, an overlay on a scanpy axes lands on the
    wrong coordinates -- say so, and name the key to pass scanpy."""
    import warnings

    x_key, bare = f"X_{basis}", str(basis)
    if x_key in adata.obsm and bare in adata.obsm:
        a, b = np.asarray(adata.obsm[x_key]), np.asarray(adata.obsm[bare])
        if a.shape != b.shape or not np.array_equal(a, b):
            warnings.warn(
                f"obsm[{x_key!r}] and obsm[{bare!r}] differ, and the velocity grid uses "
                f"{x_key!r}. Pass basis={x_key!r} to scanpy too.",
                UserWarning, stacklevel=3,
            )


def _scatter_values(ax, coords, values, *, cmap, vmin, vmax, size, na_color, zorder=2,
                    **kwargs):
    """Scatter cells colored by ``values`` in one collection; non-finite values are drawn
    first (so underneath) in ``na_color`` instead of vanishing (matplotlib's default "bad"
    color is transparent).

    Returns that collection, for a colorbar; ``None`` if no value is finite.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        ax.scatter(coords[:, 0], coords[:, 1], c=na_color, s=size, edgecolors="none",
                   zorder=zorder - 0.5, **kwargs)
        return None
    o = np.argsort(finite, kind="stable")                 # non-finite first
    s = np.asarray(size)[o] if np.ndim(size) > 0 else size
    return ax.scatter(coords[o, 0], coords[o, 1], c=values[o],
                      cmap=mpl.colormaps.get_cmap(cmap).with_extremes(bad=na_color),
                      vmin=vmin, vmax=vmax, s=s, edgecolors="none", zorder=zorder,
                      plotnonfinite=True, **kwargs)


def _draw_velocity(ax, adata, grid, *, kind, coord_scale, arrow_scale, arrow_width,
                   stream_density, stream_linewidth, stream_arrowsize, raw_subsample,
                   raw_seed, color, alpha, zorder, key_added):
    """The arrow layer itself (see :func:`velocity` for the parameters)."""
    quiver_kw = dict(color=color, angles="xy", scale_units="xy", scale=1.0,
                     width=arrow_width, headwidth=3.5, headlength=5, alpha=alpha,
                     zorder=zorder)
    if kind in ("quiver", "stream"):
        # The grid lives in the embedding's own units, so it takes the scatter's coord_scale.
        Xg, Yg = grid["Xg"] * coord_scale, grid["Yg"] * coord_scale
        U, V = grid["U"] * coord_scale, grid["V"] * coord_scale
        m = grid["mask"]
        if kind == "quiver":
            ax.quiver(Xg[m], Yg[m], arrow_scale * U[m], arrow_scale * V[m], **quiver_kw)
        else:
            ax.streamplot(Xg[0, :], Yg[:, 0], np.where(m, U, np.nan), np.where(m, V, np.nan),
                          density=stream_density, color=color, linewidth=stream_linewidth,
                          arrowsize=stream_arrowsize, zorder=zorder)
    else:
        raw = _require_uns(adata, f"dt_raw_arrows{_suffix(key_added)}", key_added,
                           "compute_velocity_grid")
        idx = np.where(raw["mask"])[0]
        if raw_subsample is not None:
            rng = np.random.default_rng(raw_seed)
            n_keep = (int(raw_subsample * len(idx)) if isinstance(raw_subsample, float)
                      else int(raw_subsample))
            idx = rng.choice(idx, size=min(n_keep, len(idx)), replace=False)
        ax.quiver(raw["X"][idx] * coord_scale, raw["Y"][idx] * coord_scale,
                  arrow_scale * raw["U"][idx] * coord_scale,
                  arrow_scale * raw["V"][idx] * coord_scale, **quiver_kw)


# ---------------------------------------------------------------------------
# pl.velocity
# ---------------------------------------------------------------------------
def velocity(
    adata,
    basis: str | None = None,
    *,
    ax=None,
    kind: str = "quiver",  # "quiver" | "stream" | "raw"
    coord_scale: float = 1.0,
    arrow_scale: float = 1.0,  # quiver / raw only
    arrow_width: float = 0.005,  # quiver / raw only
    stream_density: float = 1.5,  # stream only
    stream_linewidth: float = 1.0,  # stream only
    stream_arrowsize: float = 1.0,  # stream only
    raw_subsample=None,  # raw only: float (0-1) fraction or int count
    raw_seed: int | None = None,  # raw only: RNG seed for subsampling
    color: str = "black",
    alpha: float = 0.9,
    zorder: float = 4,
    key_added: str | None = None,
    save: str | None = None,
    show: bool = False,
) -> tuple:
    """Draw the velocity vector field on a matplotlib axes.

    Renders the velocity field computed by :func:`deltatrace.tl.compute_velocity_grid`
    as grid quivers, streamlines, or individual cell velocity vectors. Can be
    overlaid onto an existing scatter plot (e.g. from `scanpy.pl.embedding`).

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData holding ``uns["dt_grid"]`` from :func:`deltatrace.tl.compute_velocity_grid`.
    basis : str, optional
        Embedding basis to draw on. Must match the basis used for ``dt_grid``.
    ax : matplotlib.axes.Axes, optional
        Target axes to draw on. Created if None.
    kind : {"quiver", "stream", "raw"}, default "quiver"
        Style of velocity visualization: regular grid quiver arrows, continuous
        streamlines, or individual cell vectors.
    coord_scale : float, default 1.0
        Coordinate multiplier matching underlying scatter plot.
    arrow_scale : float, default 1.0
        Length multiplier for quiver/raw arrows.
    arrow_width : float, default 0.005
        Shaft width for quiver arrows.
    stream_density : float, default 1.5
        Density parameter for streamlines (when ``kind="stream"``).
    stream_linewidth : float, default 1.0
        Linewidth for streamlines.
    stream_arrowsize : float, default 1.0
        Arrowhead size for streamlines.
    raw_subsample : float or int, optional
        Subsampling fraction or count for ``kind="raw"``.
    raw_seed : int, optional
        RNG seed for raw arrow subsampling.
    color : str, default "black"
        Color of velocity arrows/streamlines.
    alpha : float, default 0.9
        Opacity of arrows.
    zorder : float, default 4
        Matplotlib drawing z-order.
    key_added : str, optional
        Namespace suffix matching :func:`deltatrace.tl.qrot`.
    save : str, optional
        File path to save the figure.
    show : bool, default False
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` when ``show=False``; ``None`` when ``show=True``.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> gr = dt.tl.endpoints(adata, ["progenitor"], ["fate_A", "fate_B"])
    >>> dt.tl.qrot(gr, adata=adata)
    >>> dt.tl.compute_velocity_grid(adata, basis="spatial")
    >>> fig, ax = dt.pl.velocity(adata, basis="spatial", kind="quiver", show=False)
    """
    if kind not in ("quiver", "stream", "raw"):
        raise ValueError(f"kind must be 'quiver', 'stream' or 'raw'; got {kind!r}")

    sfx = _suffix(key_added)
    grid_key = f"dt_grid{sfx}"
    grid = _require_uns(adata, grid_key, key_added, "compute_velocity_grid")
    grid_basis = str(grid["basis"])
    if basis is not None and str(basis) not in (grid_basis, f"X_{grid_basis}"):
        raise ValueError(
            f"The velocity grid was computed on basis={grid_basis!r}, not {basis!r}: run "
            f"tl.compute_velocity_grid(adata, basis={basis!r}) first."
        )

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_aspect("equal")
        ax.axis("off")
    else:
        fig = ax.get_figure()
        # Someone else's axes: it may have been drawn by scanpy on the other obsm key,
        # unless the caller named the X_ key the grid used.
        if str(basis) != f"X_{grid_basis}":
            _warn_if_scanpy_resolves_basis_differently(adata, grid_basis)

    _draw_velocity(ax, adata, grid, kind=kind, coord_scale=coord_scale,
                   arrow_scale=arrow_scale, arrow_width=arrow_width,
                   stream_density=stream_density, stream_linewidth=stream_linewidth,
                   stream_arrowsize=stream_arrowsize, raw_subsample=raw_subsample,
                   raw_seed=raw_seed, color=color, alpha=alpha, zorder=zorder,
                   key_added=key_added)

    if save is not None:
        _root_fig(ax).savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return _PlotResult(fig, ax)


# ---------------------------------------------------------------------------
# pl.pseudotime
# ---------------------------------------------------------------------------
def pseudotime(
    adata,
    basis: str | None = None,
    coord_scale: float = 1.0,
    cmap: str = "coolwarm",
    arrows: str = "quiver",  # "quiver" | "stream" | "raw" | "none"
    arrow_scale: float = 1.0,  # quiver / raw only
    arrow_width: float = 0.005,  # quiver / raw only
    stream_density: float = 1.5,  # stream only
    stream_linewidth: float = 1.0,  # stream only
    stream_arrowsize: float = 1.0,  # stream only
    raw_subsample=None,  # raw only: float (0-1) fraction or int count
    raw_seed: int | None = None,  # raw only: RNG seed for subsampling
    node_size: float | None = None,
    show_edges: bool = False,
    edge_alpha: float = 0.15,
    edge_width: float = 0.3,
    ax=None,
    save: str | None = None,
    show: bool = True,
    key_added: str | None = None,
    *,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    na_color: str = "lightgray",
    title: str | None = None,
) -> tuple:
    """Scatter cells colored by pseudotime, with velocity vector overlay.

    Plots cellular pseudotime on the specified coordinate embedding, optionally
    drawing regular grid quivers, streamlines, or raw cell velocity vectors.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData holding ``obs["pseudotime"]`` (from :func:`deltatrace.tl.qrot`) and
        optionally ``uns["dt_grid"]`` (from :func:`deltatrace.tl.compute_velocity_grid`).
    basis : str, optional
        Embedding coordinate name (e.g. ``"spatial"`` or ``"umap"``).
    coord_scale : float, default 1.0
        Coordinate multiplier.
    cmap : str, default "coolwarm"
        Colormap for pseudotime.
    arrows : {"quiver", "stream", "raw", "none"}, default "quiver"
        Velocity arrow style: grid quiver, streamlines, raw cell vectors, or none.
    arrow_scale, arrow_width, stream_density, stream_linewidth, stream_arrowsize, raw_subsample, raw_seed
        Styling parameters forwarded to :func:`velocity`.
    node_size : float, optional
        Marker area for cell scatter points. Defaults to Scanpy's ``120000 / n_obs``.
    show_edges : bool, default False
        Whether to draw flow graph edges underneath cells.
    edge_alpha, edge_width : float
        Styling for graph edges when ``show_edges=True``.
    ax : matplotlib.axes.Axes, optional
        Target axes to draw on.
    save : str, optional
        File path to save the figure.
    show : bool, default True
        Whether to call ``plt.show()``.
    key_added : str, optional
        Namespace suffix matching :func:`deltatrace.tl.qrot`.
    vmin, vmax : float or None, default 0.0, 1.0
        Colormap limits for pseudotime.
    colorbar : bool, default True
        Whether to draw a colorbar.
    colorbar_label : str, optional
        Label for the colorbar.
    na_color : str, default "lightgray"
        Color for cells without finite pseudotime.
    title : str, optional
        Axes title.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` when ``show=False``; ``None`` when ``show=True``.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> gr = dt.tl.endpoints(adata, ["progenitor"], ["fate_A", "fate_B"])
    >>> dt.tl.qrot(gr, adata=adata)
    >>> dt.tl.compute_velocity_grid(adata, basis="spatial")
    >>> fig, ax = dt.pl.pseudotime(adata, basis="spatial", arrows="quiver", show=False)
    """
    if arrows not in ("quiver", "stream", "raw", "none"):
        raise ValueError(
            f"arrows must be 'quiver', 'stream', 'raw', or 'none'; got {arrows!r}"
        )

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.get_figure()

    sfx = _suffix(key_added)
    grid_key = f"dt_grid{sfx}"
    if basis is None:
        basis = _require_uns(adata, grid_key, key_added, "compute_velocity_grid")["basis"]
    coords = np.asarray(_resolve_basis(adata, basis), float) * coord_scale
    pt = adata.obs[f"pseudotime{sfx}"].to_numpy(dtype=float)
    size = _auto_size(adata.n_obs) if node_size is None else node_size
    scat = _scatter_values(ax, coords, pt, cmap=cmap, vmin=vmin, vmax=vmax, size=size,
                           na_color=na_color, zorder=2)
    ax.axis("off")

    if show_edges:
        grid = _require_uns(adata, grid_key, key_added, "compute_velocity_grid")
        _draw_edges(ax, coords, grid.get("edges", []), edge_alpha, edge_width)

    if arrows != "none":
        grid = _require_uns(adata, grid_key, key_added, "compute_velocity_grid")
        _draw_velocity(ax, adata, grid, kind=arrows, coord_scale=coord_scale,
                       arrow_scale=arrow_scale, arrow_width=arrow_width,
                       stream_density=stream_density, stream_linewidth=stream_linewidth,
                       stream_arrowsize=stream_arrowsize, raw_subsample=raw_subsample,
                       raw_seed=raw_seed, color="black", alpha=0.9, zorder=4,
                       key_added=key_added)

    if colorbar and scat is not None:
        # fig.colorbar, not plt.*: a caller's ax may not be current; SubFigure has colorbar.
        cb = fig.colorbar(scat, ax=ax, shrink=0.7, pad=0.02)
        if colorbar_label:
            cb.set_label(colorbar_label)

    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)

    if save is not None:
        _root_fig(ax).savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return _PlotResult(fig, ax)

# ---------------------------------------------------------------------------
# pl.endpoints
# ---------------------------------------------------------------------------
_ENDPOINT_CLASSES = ("source", "intermediate", "sink")

# Okabe & Ito's colorblind-safe qualitative palette (yellow and black last: least legible
# as filled markers on white).
_OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442",
              "#000000")


def endpoints(
    adata,
    graph_result: dict,
    basis: str | None = None,
    coord_scale: float = 1.0,
    graph: str | None = None,
    source_color: str = "#3B4CC0",
    sink_color: str = "#B40426",
    bg_color: str = "lightgray",
    node_size_endpoint: float = 60,
    node_size_other: float | None = 10,
    show_edges: bool = True,
    edge_alpha: float = 0.15,
    edge_width: float = 0.3,
    title: str | None = None,
    save: str | None = None,
    show: bool = True,
    ax=None,
    *,
    legend: bool = True,
) -> tuple:
    """Highlight sampled source and sink endpoint cells on an embedding.

    Draws sampled source (emitter) and sink (absorber) cells highlighted over
    intermediate cells on the chosen 2D coordinate basis.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData containing coordinates in `obsm[basis]`.
    graph_result : dict
        Flow problem dictionary from :func:`deltatrace.tl.endpoints`.
    basis : str, optional
        Embedding basis to plot (e.g. ``"spatial"`` or ``"umap"``).
    coord_scale : float, default 1.0
        Coordinate multiplier.
    graph : str, optional
        Graph name to display in title. Auto-detected if None.
    source_color : str, default "#3B4CC0"
        Color for source cells.
    sink_color : str, default "#B40426"
        Color for sink cells.
    bg_color : str, default "lightgray"
        Color for intermediate background cells.
    node_size_endpoint : float, default 60
        Marker area for source/sink endpoint cells.
    node_size_other : float or None, default 10
        Marker area for intermediate cells. None uses ``120000 / n_obs``.
    show_edges : bool, default True
        Whether to draw graph edges between cells.
    edge_alpha, edge_width : float
        Edge styling parameters.
    title : str, optional
        Plot title. Defaults to ``"Endpoints: <graph>"``.
    save : str, optional
        File path to save the figure.
    show : bool, default True
        Whether to call ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Target axes to draw on.
    legend : bool, default True
        Whether to show the class legend.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` when ``show=False``; ``None`` when ``show=True``.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> gr = dt.tl.endpoints(adata, ["progenitor"], ["fate_A", "fate_B"])
    >>> fig, ax = dt.pl.endpoints(adata, gr, basis="spatial", show=False)
    """
    from matplotlib.lines import Line2D

    G_res = graph_result.get("G")
    if graph is None and isinstance(G_res, CellGraph):
        graph = G_res.graph_name              # the graph the endpoints were sampled on
    graph = _resolve_graph(adata, graph)

    G = graph_result["G"]
    source_idx = np.asarray(graph_result["source_indices"], dtype=np.int64)
    sink_idx = np.asarray(graph_result["sink_indices"], dtype=np.int64)

    if basis is None:
        grid = adata.uns.get("dt_grid")
        basis = (grid.get("basis") if isinstance(grid, dict) and grid.get("basis")
                 else next((b for b in ("spatial", "umap")
                            if f"X_{b}" in adata.obsm or b in adata.obsm), None))
    try:
        if basis is None:
            raise KeyError("no embedding")
        coords = np.asarray(_resolve_basis(adata, basis), float) * coord_scale
    except KeyError:
        # No such embedding: lay the graph out instead.
        warnings.warn(
            f"No embedding {basis!r} in obsm: using a spring layout, which is slow on "
            "large data. Pass basis= to choose an embedding.",
            UserWarning, skip_file_prefixes=(_PKG_DIR,),
        )
        G_nx = G.to_networkx() if isinstance(G, CellGraph) else G   # layout is networkx
        pos = nx.spring_layout(G_nx.to_undirected(), seed=42)
        coords = np.array([pos[i] for i in range(adata.n_obs)]) * coord_scale

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.get_figure()

    ep_class = np.full(adata.n_obs, "intermediate", dtype=object)
    ep_class[source_idx] = "source"
    ep_class[sink_idx] = "sink"
    class_colors = {"source": source_color, "intermediate": bg_color, "sink": sink_color}
    size_other = _auto_size(adata.n_obs) if node_size_other is None else node_size_other

    # Every cell small, in its class color, so the picture reads even without the overlay
    for cls in _ENDPOINT_CLASSES:
        sel = np.flatnonzero(ep_class == cls)
        if sel.size:
            ax.scatter(coords[sel, 0], coords[sel, 1], c=class_colors[cls], s=size_other,
                       edgecolors="none", zorder=2)

    # The sampled endpoints again, large, on top
    for idx, color in ((source_idx, source_color), (sink_idx, sink_color)):
        if idx.size:
            ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=node_size_endpoint,
                       alpha=0.9, zorder=3, edgecolors="none")

    if show_edges:
        edges = G.edge_array if isinstance(G, CellGraph) else np.array(list(G.edges()))
        _draw_edges(ax, coords, edges, edge_alpha, edge_width)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title if title is not None else f"Endpoints: {graph}")

    if legend:
        # Proxy handles in a fixed order; source-only runs simply have no 'sink' entry.
        present = [c for c in _ENDPOINT_CLASSES if (ep_class == c).any()]
        ax.legend(
            [Line2D([], [], marker="o", linestyle="", color=class_colors[c], label=c)
             for c in present],
            present,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            borderaxespad=0,
            framealpha=0.6,
        )

    if save is not None:
        _root_fig(ax).savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return _PlotResult(fig, ax)

# ---------------------------------------------------------------------------
# pl.lineages
# ---------------------------------------------------------------------------
def lineages(
    adata,
    basis: str | None = None,
    lineage_colors: dict | None = None,
    key_added: str | None = None,
    cmap: str = "okabe_ito",
    unassigned_color: str = "lightgray",
    confidence_alpha: bool = True,
    alpha_range: tuple = (0.25, 1.0),
    coord_scale: float = 1.0,
    node_size: float | None = None,
    unassigned_size: float | None = None,
    legend: bool = True,
    title: str | None = None,
    ax=None,
    save: str | None = None,
    show: bool = True,
) -> tuple:
    """Scatter cells colored by their committed terminal lineage.

    Colors cells by assigned terminal fate (``obs["lineage"]``), with marker opacity
    scaled by lineage confidence (``obs["lineage_confidence"]``) so multipotent or
    transitional cells appear faint and committed cells appear solid.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData holding ``obs["lineage"]``, ``obs["lineage_confidence"]`` and
        ``uns["path_lineages"]`` from :func:`deltatrace.tl.lineages`.
    basis : str, optional
        Embedding coordinate name. Defaults to the basis in ``uns["dt_grid"]``.
    lineage_colors : dict, optional
        Explicit mapping of ``{lineage_name: color}``.
    key_added : str, optional
        Namespace suffix matching :func:`deltatrace.tl.lineages`.
    cmap : str, default "okabe_ito"
        Colormap or palette name for lineage colors (defaults to Okabe-Ito).
    unassigned_color : str, default "lightgray"
        Color for cells not reaching any terminal fate.
    confidence_alpha : bool, default True
        Whether to modulate marker alpha by lineage confidence.
    alpha_range : tuple of float, default (0.25, 1.0)
        Minimum and maximum alpha when confidence modulation is active.
    coord_scale : float, default 1.0
        Coordinate multiplier.
    node_size : float, optional
        Marker area for assigned cells. Defaults to ``120000 / n_obs``.
    unassigned_size : float, optional
        Marker area for unassigned cells.
    legend : bool, default True
        Whether to show the lineage color legend.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Target axes to draw on.
    save : str, optional
        File path to save the figure.
    show : bool, default True
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` when ``show=False``; ``None`` when ``show=True``.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> gr = dt.tl.endpoints(adata, ["progenitor"], ["fate_A", "fate_B"])
    >>> dt.tl.qrot(gr, adata=adata)
    >>> lin = dt.tl.lineages(adata, ["progenitor"], ["fate_A", "fate_B"])
    >>> fig, ax = dt.pl.lineages(adata, basis="spatial", show=False)
    """
    sfx = _suffix(key_added)
    lin_key, conf_key = f"lineage{sfx}", f"lineage_confidence{sfx}"
    if lin_key not in adata.obs:
        raise KeyError(
            f"obs['{lin_key}'] not found: run deltatrace.tl.lineages"
            + (f" with key_added={key_added!r}" if key_added else "")
            + " first."
        )
    names = [str(n) for n in adata.uns[f"path_lineages{sfx}"]]

    if basis is None:
        grid = adata.uns.get(f"dt_grid{sfx}")
        if grid is None or "basis" not in grid:
            raise KeyError(
                "basis is None and no uns['dt_grid'] to read it from: pass basis= "
                "explicitly (e.g. basis='spatial')."
            )
        basis = grid["basis"]
    coords = _resolve_basis(adata, basis) * coord_scale

    # Resolve one color per lineage: colormap first, then the caller's overrides.
    if cmap == "okabe_ito":
        cmap = "okabe_ito" if len(names) <= len(_OKABE_ITO) else "tab20"
    if cmap == "okabe_ito":
        palette = dict(zip(names, _OKABE_ITO, strict=False))
    elif hasattr(mpl.colormaps[cmap], "colors"):
        palette = dict(zip(names, itertools.cycle(mpl.colormaps[cmap].colors)))
    else:
        palette = {n: mpl.colormaps[cmap](i / max(len(names) - 1, 1)) for i, n in enumerate(names)}
    if lineage_colors:
        unknown = [k for k in lineage_colors if str(k) not in names]
        if unknown:
            raise ValueError(
                f"lineage_colors names {unknown} which are not lineages of this run; "
                f"available: {names}"
            )
        palette.update({str(k): v for k, v in lineage_colors.items()})

    lab = adata.obs[lin_key].astype(str).to_numpy()
    if confidence_alpha and conf_key in adata.obs:
        conf = adata.obs[conf_key].to_numpy(float)
        lo, hi = alpha_range
        finite = np.isfinite(conf)
        span = (np.nanmax(conf[finite]) - np.nanmin(conf[finite])) if finite.any() else 0.0
        if span > 0:
            a = lo + (conf - np.nanmin(conf[finite])) / span * (hi - lo)
        else:
            a = np.full(conf.shape, hi)
        alpha_vec = np.where(finite, a, lo)
    else:
        alpha_vec = np.full(lab.shape, alpha_range[1], dtype=float)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.get_figure()

    if node_size is None:
        node_size = _auto_size(adata.n_obs)

    # Draw unassigned cells first so committed cells are never hidden underneath them
    un = lab == "unassigned"
    if un.any():
        ax.scatter(coords[un, 0], coords[un, 1], c=unassigned_color,
                   s=unassigned_size if unassigned_size is not None else node_size,
                   edgecolors="none", zorder=1)
    drawn = []
    for name in names:
        m = lab == name
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], color=palette[name], s=node_size,
                   alpha=alpha_vec[m], edgecolors="none", zorder=2)
        drawn.append(name)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if legend and (drawn or un.any()):
        # Proxy handles show each lineage's color at full opacity, which per-cell alpha cannot.
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], marker="o", linestyle="", color=palette[n], label=n)
                   for n in drawn]
        if un.any():
            handles.append(Line2D([], [], marker="o", linestyle="",
                                  color=unassigned_color, label="unassigned"))
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
                  borderaxespad=0, framealpha=0.6)

    if save is not None:
        _root_fig(ax).savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return _PlotResult(fig, ax)


# ---------------------------------------------------------------------------
# pl.cluster_trajectory
# ---------------------------------------------------------------------------
def _graph_layout(res, n, layout, seed, kwds=None):
    """Node positions from the cluster GRAPH itself, ignoring any cell embedding.

    A cell embedding places a node at its cluster's median coordinate, so on UMAP the
    edges can cross unrelated regions -- UMAP is not metric between clusters. Laying the
    coarse graph out on its own is what makes a backbone readable, and is what
    ``sc.pl.paga`` does by default.
    """
    import networkx as nx

    fns = {"kamada_kawai": nx.kamada_kawai_layout, "spring": nx.spring_layout,
           "fr": nx.spring_layout,          # spring_layout IS Fruchterman-Reingold
           "forceatlas2": getattr(nx, "forceatlas2_layout", None),
           "spectral": nx.spectral_layout, "circular": nx.circular_layout,
           "planar": nx.planar_layout, "shell": nx.shell_layout}
    if layout != "pseudotime" and (layout not in fns or fns[layout] is None):
        avail = [*sorted(k for k, v in fns.items() if v is not None), "pseudotime"]
        raise ValueError(f"layout must be one of {avail} or None, got {layout!r}")

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(r.i), int(r.j)) for r in res["edges"].itertuples())

    if layout == "pseudotime":
        # x IS the cluster pseudotime, so position encodes the method's own output:
        # left-to-right is time and branches separate vertically. No nx layout reads a
        # continuous node value -- multipartite_layout only takes discrete layers -- so
        # y comes from Kamada-Kawai and x is overwritten.
        pt = np.asarray(res["cluster_pt"], float)
        if not np.isfinite(pt).any():
            raise ValueError("layout='pseudotime' needs finite cluster_pt; none present.")
        lo, hi = np.nanmin(pt[np.isfinite(pt)]), np.nanmax(pt[np.isfinite(pt)])
        x = (pt - lo) / (hi - lo) if hi > lo else np.zeros_like(pt)
        x = np.where(np.isfinite(x), x, 0.0)
        pos = nx.kamada_kawai_layout(G)
        y = np.array([pos[i][1] for i in range(n)], float)
        return np.column_stack([x, y])

    fn = fns[layout]
    kw = dict(kwds or {})
    try:
        pos = fn(G, seed=seed, **kw)
    except TypeError:                       # kamada_kawai / spectral take no seed
        pos = fn(G, **kw)
    return np.array([pos[i] for i in range(n)], dtype=float)


def _span_norm(v, lo, hi):
    """Map values onto ``[lo, hi]`` by their own range; a constant input (no range to
    normalize by) is put at the midpoint instead of collapsing to ``lo``."""
    v = np.asarray(v, float)
    span = float(v.max() - v.min()) if v.size else 0.0
    if not np.isfinite(span) or span <= 1e-12:
        return np.full(v.shape, (lo + hi) / 2.0)
    return lo + (v - v.min()) / span * (hi - lo)


def cluster_trajectory(
    adata,
    basis: str | None = None,
    *,
    cluster_key: str = "clusters",
    key_added: str | None = None,
    cmap: str = "coolwarm",
    edge_lw_range: tuple = (0.8, 4.0),
    node_size_range: tuple = (150, 1400),
    arrow_size: float = 0.04,
    node_pad: float = 0.15,
    flag_flipped: bool = True,
    layout: str | None = None,
    layout_seed: int = 0,
    layout_kwds: dict | None = None,
    title: str | None = None,
    save: str | None = None,
    show: bool = True,
    ax=None,
) -> tuple:
    """Draw the coarse-grained cluster trajectory graph directed by net transport.

    Renders clusters as nodes sized by transported mass and colored by median
    pseudotime, with directed arrows weighted by net inter-cluster transport flux.
    Nodes can be positioned using cell embedding coordinates (`basis`) or an
    abstract graph layout algorithm (`layout`).

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData holding ``uns["dt_cluster_trajectory"]`` from :func:`deltatrace.tl.cluster_trajectory`.
    basis : str, optional
        Embedding name for node positions (defaults to the basis in ``uns["dt_grid"]``).
    cluster_key : str, default "clusters"
        Column in ``adata.obs`` with cluster labels matching the cluster graph.
    key_added : str, optional
        Namespace suffix matching :func:`deltatrace.tl.cluster_trajectory`.
    cmap : str, default "coolwarm"
        Colormap for cluster median pseudotime.
    edge_lw_range, node_size_range : tuple of float
        Display ranges for edge linewidths and node marker sizes.
    arrow_size : float, default 0.04
        Arrowhead size relative to edge length.
    node_pad : float, default 0.15
        Axis padding fraction around nodes.
    flag_flipped : bool, default True
        Whether to draw dashed lines for edges flipped by the acyclicity guard.
    layout : str, optional
        Abstract network layout instead of embedding coordinates (e.g. ``"kamada_kawai"``,
        ``"spring"``, or ``"pseudotime"``).
    layout_seed : int, default 0
        Random seed for stochastic graph layouts.
    layout_kwds : dict, optional
        Additional keyword arguments passed to the networkx layout function.
    title : str, optional
        Plot title.
    save : str, optional
        File path to save the figure.
    show : bool, default True
        Whether to call ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Target axes to draw on.

    Returns
    -------
    tuple or None
        ``(fig, ax)`` when ``show=False``; ``None`` when ``show=True``.

    Examples
    --------
    >>> import deltatrace as dt
    >>> adata, adata_dist = dt.datasets.crypt()
    >>> adata = dt.pp.integrate(adata, adata_dist=adata_dist)
    >>> gr = dt.tl.endpoints(adata, ["progenitor"], ["fate_A", "fate_B"])
    >>> dt.tl.qrot(gr, adata=adata)
    >>> dt.tl.cluster_trajectory(adata)
    >>> fig, ax = dt.pl.cluster_trajectory(adata, basis="spatial", show=False)
    """
    import matplotlib.colors as mcolors

    sfx = _suffix(key_added)
    res = _require_uns(adata, f"dt_cluster_trajectory{sfx}", key_added,
                       "cluster_trajectory")

    coords = None
    if layout is None:
        if basis is None:
            grid = adata.uns.get(f"dt_grid{sfx}")
            if grid is None or "basis" not in grid:
                raise ValueError(
                    f"basis is None and uns['dt_grid{sfx}'] records none: pass basis= "
                    "(e.g. 'umap') or layout=."
                )
            basis = grid["basis"]
        coords = np.asarray(_resolve_basis(adata, basis), float)

    # Reject a column other than the one the nodes were built from: the nodes are
    # positions in a fixed list of cluster names, so a different column reusing those
    # names would silently place every node on the wrong cells.
    graph_key = res.get("cluster_key")
    if graph_key is not None and str(graph_key) != str(cluster_key):
        raise ValueError(
            f"cluster_key={cluster_key!r}, but the cluster trajectory was computed on "
            f"{str(graph_key)!r}: pass cluster_key={str(graph_key)!r}, or rerun "
            f"tl.cluster_trajectory on {cluster_key!r}."
        )

    names = [str(c) for c in res["names"]]
    labels = adata.obs[cluster_key].astype(str)     # KeyError here is correct: bad column
    present = set(labels.unique())
    missing = [n for n in names if n not in present]
    if missing:
        raise ValueError(
            f"The cluster trajectory's clusters {missing[:5]} are not in "
            f"obs['{cluster_key}']: pass the cluster_key it was computed on, or rerun "
            "tl.cluster_trajectory."
        )

    # Node position: the MEDIAN of the cluster's coordinates, not the mean, since
    # reconnection-backfilled cells sit wherever the tissue put them and can drag a
    # mean centroid outside the cluster entirely.
    if layout is None:
        node_pos = (pd.DataFrame(coords).groupby(labels.to_numpy(), sort=False).median()
                    .loc[names].to_numpy())
    else:
        node_pos = _graph_layout(res, len(names), layout, layout_seed, layout_kwds)

    node_sizes = _span_norm(res["throughput"], *node_size_range)

    pt_vals = np.asarray(res["cluster_pt"], float)
    cmap_fn = mpl.colormaps[cmap]
    if np.isfinite(pt_vals).any():
        finite = pt_vals[np.isfinite(pt_vals)]
        norm = mcolors.Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
        node_colors = cmap_fn.with_extremes(bad="lightgray")(norm(np.ma.masked_invalid(pt_vals)))
    else:                                   # no pseudotime column: one neutral color
        node_colors = ["lightgray"] * len(names)

    edges = res["edges"]
    lws = _span_norm(edges["weight"].to_numpy(float), *edge_lw_range)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # One line per retained edge, with a midpoint arrow pointing the way the mass moves.
    for row, lw in zip(edges.itertuples(index=False), lws, strict=True):
        p1, p2 = node_pos[int(row.i)], node_pos[int(row.j)]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="gray",
            lw=lw,
            alpha=0.6,
            zorder=1,
            solid_capstyle="round",
            linestyle="--" if (flag_flipped and row.flipped) else "-",
        )

        mid = (p1 + p2) / 2.0
        delta = p2 - p1
        length = float(np.linalg.norm(delta))
        if length < 1e-12:
            continue
        unit = delta / length
        eps = arrow_size * length
        ax.annotate(
            "",
            xy=mid + eps * unit,
            xytext=mid - eps * unit,
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=max(lw * 0.7, 0.8),
                mutation_scale=12,
            ),
            zorder=2,
        )

    ax.scatter(
        node_pos[:, 0],
        node_pos[:, 1],
        c=node_colors,
        s=node_sizes,
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )
    for i, name in enumerate(names):
        ax.text(
            node_pos[i, 0],
            node_pos[i, 1],
            name,
            ha="center",
            va="center",
            fontsize=7,
            zorder=4,
        )

    # Pad the axes around the nodes so markers and labels are not clipped
    x_span = max(node_pos[:, 0].max() - node_pos[:, 0].min(), 1e-6)
    y_span = max(node_pos[:, 1].max() - node_pos[:, 1].min(), 1e-6)
    ax.set_xlim(
        node_pos[:, 0].min() - node_pad * x_span,
        node_pos[:, 0].max() + node_pad * x_span,
    )
    ax.set_ylim(
        node_pos[:, 1].min() - node_pad * y_span,
        node_pos[:, 1].max() + node_pad * y_span,
    )
    ax.set_aspect("equal", adjustable="box")    # keep the padded limits; the box fits them
    ax.axis("off")

    if title:
        ax.set_title(title)
    # Only lay out a figure we created (a caller's may be a SubFigure, or hold other panels).
    if owns_fig:
        fig.tight_layout()
    if save:
        # _root_fig: on an axes from fig.subfigures(), `fig` is a SubFigure with no savefig.
        _root_fig(ax).savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return _PlotResult(fig, ax)
