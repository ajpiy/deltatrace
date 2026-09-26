"""Synthetic datasets (``deltatrace.datasets``): the example system for docs and tutorials.

Holds :func:`crypt`, which hands back a ready-to-use pair of objects for the
:func:`deltatrace.pp.graphs` entry point, synthesized deterministically from ``seed``: a
crypt-shaped tissue whose two walls run the same program, so that tissue distance changes
the answer.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

__all__ = ["crypt"]


# crypt(): stage cuts, as fractions of the way from the bottom of the U to a wall's top.
_CRYPT_PROGENITOR_END = 0.25  # the base (progenitors) gives way to transit cells
_CRYPT_FATE_START = 0.85  # transit gives way to fate_A / fate_B
# crypt(): fixed expression settings.
_CRYPT_FORK_SIZE = 0.7  # level marker_A / marker_B reach at the top (stage genes span 0-1)
_CRYPT_BASELINE = 0.2  # added to every gene, so expression stays non-negative
_CRYPT_PROGRAM_NOISE = 0.5  # noise on the four program genes, as a fraction of expr_noise
_CRYPT_N_NOISE_GENES = 4
_CRYPT_N_COMPS = 4
_CRYPT_FALSE_FATE_AT = (0.55, 0.65)  # stage (height) of the false-fate patch on the left wall
_CRYPT_FALSE_FATE_AS = (1.0, 1.1)  # the fate_B stage it expresses: just past that wall's top


def crypt(
    n: int = 150,
    *,
    seed: int = 15,
    lumen_width: float = 5.0,
    wall_height: float = 7.0,
    wall_thickness: float = 1.0,
    fork_start: float = 0.7,
    expr_noise: float = 0.04,
    false_fate: float = 0.0,
    n_neighbors: int = 20,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Simulate a crypt-shaped tissue whose two walls run the same maturation program.

    ``n`` cells line a U, like a crypt in section: progenitors at the curved base, and
    two straight walls rising from it on either side of an empty lumen. A cell's stage
    ``t`` in ``[0, 1]`` is how far it sits along its own wall from the bottom of the U,
    and expression reads the stage out the same way on both walls (``stem`` falls,
    ``mature`` rises), so cells at the same height on opposite walls look alike. Only
    near the top do the walls diverge: from ``fork_start`` on, the left wall switches on
    ``marker_A`` and the right wall ``marker_B``. Expression is therefore a line with a
    short fork at its end (``obsm["X_expr2d"]``, a Y), and the tissue is a U
    (``obsm["spatial"]``).

    Clusters are cut from the stage: ``"progenitor"`` (``t < 0.25``, the base, shared by
    both walls), ``"transit"`` (one cluster spanning both walls, since expression cannot
    split it) and ``"fate_A"`` / ``"fate_B"`` (``t >= 0.85``, the top of the left / right
    wall). Which top a transit cell reaches is set by its wall, which expression cannot
    see below the fork and the tissue can. The returned expression object already
    carries a PCA and a kNN graph, so it can go straight into
    :func:`deltatrace.pp.graphs`, with ``["progenitor"]`` as the source and
    ``["fate_A", "fate_B"]`` as the sinks.

    The expression kNN graph is a ladder: besides the links along each wall, it has
    rungs straight across the lumen between cells of the same stage. With the default
    geometry, the ``p_keep=0.32`` tissue radius of :func:`deltatrace.pp.graphs` comes out
    near 4 units, shorter than the 5-unit lumen, so the intersection graph (``"intr"``)
    cuts every rung. The walls stay joined through the base, which is one continuous
    stretch of tissue whose cells are alike in expression, so ``"intr"`` is one
    connected U. On it the flow runs up each wall and each wall's cells go to their own
    top. On the expression graph the flow jumps the lumen, and most mid-wall cells are
    left undecided between the two tops. The rungs come back once ``p_keep`` pushes the
    radius past the lumen, just above ``p_keep=0.4``.

    Parameters
    ----------
    n : int, default 150
        Number of cells, at least 40. They are split evenly between the two walls and
        spread evenly, with jitter, along each. ``150`` keeps the tests fast and the
        quickstart uses ``300``; ``1000`` gives a dense picture and still runs through
        :func:`deltatrace.pp.graphs` and :func:`deltatrace.tl.trajectory` in under a
        second.
    seed : int, default 15
        Seed for the placement jitter, the expression noise and the cell order. Every
        seed teaches the same lesson; the default is a draw on which the two walls'
        pseudotime clocks agree (see Notes).
    lumen_width : float, default 5.0
        Gap between the inner faces of the two walls, in tissue units.
    wall_height : float, default 7.0
        Height of each straight wall above the curved base, in tissue units.
    wall_thickness : float, default 1.0
        Thickness of the cell layer, in tissue units. Cells scatter uniformly across it.
    fork_start : float, default 0.7
        Stage from which the walls' expression diverges; ``marker_A`` / ``marker_B`` then
        rise linearly to ``0.7`` at the top. ``1.0`` gives no fork: a pure line, whose
        two tops differ only in where they sit. ``0.0`` makes the walls differ from the
        base up, a control in which expression alone already tells them apart.
    expr_noise : float, default 0.04
        Standard deviation of the Gaussian noise on the noise genes. The four program
        genes (``stem``, ``mature``, ``marker_A``, ``marker_B``) each stand for a whole
        gene program and carry half of it.
    false_fate : float, default 0.0
        Fraction of ``n`` (at least 3 cells when positive) placed as a ``"false_fate"``
        patch half-way up the left wall that expresses the ``fate_B`` program just past
        that fate's end: an end state no lineage leads to. Expression alone takes it for a
        terminal state (so source-only discovery on expression is fooled by it), while ``"intr"`` cuts it off as a component of its own, because the cells
        it resembles sit across the lumen. ``0.0`` leaves it out. Needs ``n >= 150``
        (warns otherwise): in smaller data a patch this size cannot fill its own
        ``n_neighbors`` expression neighbors, so some land on its own wall.
    n_neighbors : int, default 20
        Neighbors for :func:`scanpy.pp.neighbors`. About half of each cell's expression
        neighbors are rungs across the lumen, which the intersection drops, so the
        default sits above scanpy's 15: at 15 a wall top breaks off ``"intr"`` in about
        one dataset in 150 at ``n`` of 150-300; at 20 it never did in 27,000 datasets
        with ``n`` from 40 to 1000.

    Returns
    -------
    adata : anndata.AnnData
        ``n`` cells by 8 non-negative genes (``stem``, ``mature``, ``marker_A``,
        ``marker_B``, ``noise0``-``noise3``), with ``obs["clusters"]``, ``obs["wall"]``
        (``"base"`` for the progenitors, else ``"left"`` / ``"right"``),
        ``obs["true_time"]`` (the stage ``t``; NaN for the false fate), ``obs["true_fate"]``
        (``"source"`` for progenitors, ``"none"`` for the false fate, else the fate at the
        top of the cell's wall), ``obsm["spatial"]`` (tissue units, base at the bottom),
        ``obsm["X_expr2d"]`` (x: ``marker_B - marker_A``, y: the maturation score
        ``(mature - stem + 1) / 2``), ``obsm["X_pca"]``, an expression kNN graph under
        the default ``"neighbors"`` key, ``uns["story"]`` (the source, sink and
        false-fate clusters) and ``uns["crypt"]`` (the geometry and the stage cuts, for
        drawing the lumen against the ``p_keep`` radius). Cells are stored in random
        order.
    adata_dist : anndata.AnnData
        Dense ``(n, n)`` pairwise tissue distance matrix, in the shape
        :func:`deltatrace.pp.graphs` expects for its ``adata_dist`` argument, with both
        axes labeled by ``adata.obs_names``.

    Notes
    -----
    The crypt lends the example only its shape: cells of the same stage face each other
    across an empty lumen, so expression neighbors need not be tissue neighbors. It is
    not a model of crypt biology. A real crypt is a tube whose section's two walls are
    one epithelium, and its fates are not wall-specific; "the wall decides the fate" is
    built in to give the example two sinks.

    Lengths are in nominal tissue units, with the cell layer 1 unit thick, so the U is
    about 7 units wide and 10.5 tall. That size keeps
    the velocity arrows of :func:`deltatrace.pl.pseudotime`, which are drawn in data
    units, at a legible length. Scaling every length together changes no result,
    because :func:`deltatrace.pp.graphs` sets its radius as a quantile of all pairwise
    distances. Changing one length does: the radius is a quantile over the whole tissue,
    so taller walls, a narrower lumen or thicker walls can push it past the lumen and
    bring the rungs back (``wall_height=11`` or ``lumen_width=3`` already lets some flow
    cross).

    Pseudotime is not where the tissue helps. Within each wall it tracks ``t`` about
    equally well on both graphs. But on ``"intr"`` each wall keeps only its own
    expression links and runs its own clock, and the transport's congestion term
    (``tau`` in :func:`deltatrace.tl.trajectory`) magnifies small differences between
    the two walls' links. The clocks therefore drift apart, so cells at the same height
    on opposite walls get different pseudotime: by about 0.06 on average at ``n=150``
    and 0.03 at ``n=1000``. The expression graph mixes the walls, so they share one
    clock. Read on one clock across both walls, DeltaTrace therefore loses a little at
    small ``n`` (Spearman with ``t`` about 0.96 against 0.99 at ``n=150``) and nearly
    ties at ``n=1000``.

    At package defaults ``"intr"`` was one component in every one of 27,000 draws with
    ``n`` from 40 to 1000. From about ``n=1200`` a single transit cell falls off it in
    about one draw in 150: a cell just past ``fork_start`` whose noise shows the other
    wall's marker, so that all its expression neighbors sit across the lumen.

    Some cells receive no flow and stay ``"unassigned"``, more of them as ``n`` grows. The
    wall tips beyond the centered sink cells do so on ``"intr"`` (at ``n=1000`` a median of
    about a fifth of the ``fate_A`` / ``fate_B`` cells, fewer at smaller ``alpha``). On
    some draws from about ``n=600``, the very bottom of the base does so on both graphs:
    the source cells are centered on the progenitor cluster's best-connected cells, which
    sit on the flanks of the base, so the bottom of the U lies behind them.
    """
    import scanpy as sc

    for name, value in (("n", n), ("n_neighbors", n_neighbors)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"crypt(): {name} must be an integer, got {value!r}.")
    if not 0.0 <= false_fate < 0.5:
        raise ValueError(f"crypt(): false_fate must lie in [0, 0.5), got {false_fate}.")
    n_false = max(3, round(false_fate * n)) if false_fate > 0 else 0
    n_line = n - n_false
    if n_line < 40:
        raise ValueError(
            f"crypt(): need at least 40 lineage cells to line both walls, got {n_line}."
        )
    if n_neighbors < 2:
        raise ValueError(f"crypt(): n_neighbors must be at least 2, got {n_neighbors}.")
    if not 0.0 <= fork_start <= 1.0:
        raise ValueError(f"crypt(): fork_start must lie in [0, 1], got {fork_start}.")
    if min(lumen_width, wall_height, wall_thickness) <= 0:
        raise ValueError("crypt(): lumen_width, wall_height and wall_thickness must be > 0.")
    if expr_noise < 0:
        raise ValueError(f"crypt(): expr_noise must be >= 0, got {expr_noise}.")
    if n_false and n < 150:
        warnings.warn(
            f"crypt(): with n={n} the false-fate patch is too small to fill its own "
            f"{n_neighbors} expression neighbors, so some reach its own wall and 'intr' "
            f"does not cut it off. Use n >= 150.",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    n_genes = 4 + _CRYPT_N_NOISE_GENES

    # Where each lineage cell sits: stage t = arc length from the bottom of the U along its
    # own wall, as a fraction of the wall's length. Stratified (one cell per equal slot,
    # jittered inside it) so that no stretch of wall is left empty by chance.
    r_base = (lumen_width + wall_thickness) / 2  # radius of the curved base's center line
    base_len = np.pi * r_base / 2  # arc length of one half of the base
    wall_len = base_len + wall_height  # bottom of the U to the top of a wall
    n_left = n_line // 2
    side = np.r_[-np.ones(n_left), np.ones(n_line - n_left)]  # -1 left wall, +1 right
    rank = np.r_[np.arange(n_left), np.arange(n_line - n_left)]
    per_wall = np.r_[np.full(n_left, n_left), np.full(n_line - n_left, n_line - n_left)]
    t = (rank + rng.uniform(0, 1, n_line)) / per_wall
    offset = rng.uniform(-wall_thickness / 2, wall_thickness / 2, n_line)
    noise = rng.normal(0, expr_noise, (n_line, n_genes))
    # The program genes carry less noise than the noise genes: the stage order along each
    # wall stays tight, while the noise genes still spread each cell's expression
    # neighbors over a wide stretch of stage. With full noise on every gene, a wall top
    # lost all its links to the rest of its wall in about one dataset in 1000.
    noise[:, :4] *= _CRYPT_PROGRAM_NOISE
    stage, program = t.copy(), side.copy()  # what the expression shows
    if n_false:  # a patch on the LEFT wall that expresses the right wall's top
        t = np.r_[t, rng.uniform(*_CRYPT_FALSE_FATE_AT, n_false)]
        side = np.r_[side, -np.ones(n_false)]
        offset = np.r_[offset, rng.uniform(-wall_thickness / 2, wall_thickness / 2, n_false)]
        noise_false = rng.normal(0, expr_noise, (n_false, n_genes))
        noise_false[:, :4] *= _CRYPT_PROGRAM_NOISE
        noise = np.r_[noise, noise_false]
        stage = np.r_[stage, rng.uniform(*_CRYPT_FALSE_FATE_AS, n_false)]
        program = np.r_[program, np.ones(n_false)]
    is_false = np.r_[np.zeros(n_line, dtype=bool), np.ones(n_false, dtype=bool)]

    s = t * wall_len
    r_cell = r_base + offset
    theta = np.minimum(s / r_base, np.pi / 2)  # angle round the base
    y0 = r_base + wall_thickness / 2  # base center; the U's outer bottom is y=0
    coords = np.column_stack(
        [
            side * r_cell * np.sin(theta),
            np.where(s < base_len, y0 - r_cell * np.cos(theta), y0 + (s - base_len)),
        ]
    )

    # What each cell expresses: one stage program on both walls, plus a wall's own
    # terminal marker from fork_start on.
    if fork_start < 1.0:
        fork = _CRYPT_FORK_SIZE * np.clip((stage - fork_start) / (1.0 - fork_start), 0, None)
    else:
        fork = np.zeros(n)
    genes = np.column_stack(
        [
            1.0 - stage,  # stem
            stage,  # mature
            np.where(program < 0, fork, 0.0),  # marker_A (left-wall program)
            np.where(program < 0, 0.0, fork),  # marker_B (right-wall program)
            np.zeros((n, _CRYPT_N_NOISE_GENES)),
        ]
    )
    X = np.clip(genes + _CRYPT_BASELINE + noise, 0.0, None)

    left = side < 0
    base = ~is_false & (t < _CRYPT_PROGENITOR_END)
    top = np.where(left, "fate_A", "fate_B")
    clusters = np.where(
        is_false,
        "false_fate",
        np.where(base, "progenitor", np.where(t < _CRYPT_FATE_START, "transit", top)),
    )
    true_fate = np.where(is_false, "none", np.where(base, "source", top))
    wall = np.where(base, "base", np.where(left, "left", "right"))

    # Store the cells in random order (drawn last, so it moves nothing else): downstream
    # tie-breaks take the lowest index, which in build order would always be the left wall.
    order = rng.permutation(n)
    obs = pd.DataFrame(
        {
            "clusters": pd.Categorical(
                clusters[order],
                categories=["progenitor", "transit", "fate_A", "fate_B"]
                + (["false_fate"] if n_false else []),
            ),
            "wall": pd.Categorical(wall[order], categories=["base", "left", "right"]),
            "true_time": np.where(is_false, np.nan, t)[order],
            "true_fate": pd.Categorical(
                true_fate[order],
                categories=["source", "fate_A", "fate_B"] + (["none"] if n_false else []),
            ),
        },
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame(
        index=["stem", "mature", "marker_A", "marker_B"]
        + [f"noise{i}" for i in range(_CRYPT_N_NOISE_GENES)]
    )
    X = X[order].astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords[order]
    # The picture of expression space: the two contrasts the marker genes encode.
    Xd = X.astype(np.float64)
    adata.obsm["X_expr2d"] = np.column_stack([Xd[:, 3] - Xd[:, 2], (Xd[:, 1] - Xd[:, 0] + 1) / 2])
    adata.uns["story"] = {
        "source_clusters": ["progenitor"],
        "sink_clusters": ["fate_A", "fate_B"],
        "decoy_clusters": ["false_fate"] if n_false else [],
    }
    adata.uns["crypt"] = {
        "units": "tissue units",
        "lumen_width": float(lumen_width),
        "wall_height": float(wall_height),
        "wall_thickness": float(wall_thickness),
        "base_centre": [0.0, float(y0)],  # the lumen is |x| < lumen_width / 2
        "base_radius": float(r_base),  # center line of the cell layer
        "progenitor_end": _CRYPT_PROGENITOR_END,
        "fate_start": _CRYPT_FATE_START,
        "fork_start": float(fork_start),
    }

    sc.pp.pca(adata, n_comps=_CRYPT_N_COMPS)
    sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, n - 1), use_rep="X_pca")

    D = cdist(adata.obsm["spatial"], adata.obsm["spatial"])
    np.fill_diagonal(D, 0.0)
    # Label both axes with the cell names: pp.integrate matches the pair by obs_names.
    adata_dist = ad.AnnData(
        X=D.astype(np.float64),
        obs=pd.DataFrame(index=adata.obs_names.copy()),
        var=pd.DataFrame(index=adata.obs_names.copy()),
    )

    return adata, adata_dist
