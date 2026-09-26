"""The line toy the test suite pins its numbers on (formerly ``deltatrace.datasets.toy``).

The package now ships :func:`deltatrace.datasets.crypt` as its example dataset. This
``source -> mid -> sink`` continuum stays here, unchanged, because many tests use it as
a small, fully deterministic input and pin exact values on it.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

__all__ = ["line_toy"]


def line_toy(
    n: int = 150,
    *,
    seed: int = 0,
    n_noise_genes: int = 8,
    spatial_scale: float = 10.0,
    spatial_jitter: float = 0.25,
    expr_noise: float = 0.05,
    marker_noise: float = 0.05,
    n_comps: int = 5,
    n_neighbors: int = 12,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Simulate a ``source -> mid -> sink`` cell continuum with paired spatial coordinates.

    ``n`` cells are placed on a latent pseudotime ``t = linspace(0, 1, n)``, expressed as
    two opposing noisy marker genes (``t`` and ``1 - t``) plus ``n_noise_genes`` pure-noise
    genes, and laid out left-to-right in the tissue along ``t``. Cluster labels split the
    continuum in thirds: ``"source"``, ``"mid"``, ``"sink"``. The returned expression object
    already carries a PCA and a kNN graph, so it can go straight into
    :func:`deltatrace.pp.graphs`.

    Parameters
    ----------
    n : int, default 150
        Number of cells.
    seed : int, default 0
        Seed for the noise and spatial jitter.
    n_noise_genes : int, default 8
        Pure-noise genes appended after the two marker genes.
    spatial_scale : float, default 10.0
        Length of the tissue axis; coordinates run from ``0`` to ``spatial_scale``.
    spatial_jitter : float, default 0.25
        Standard deviation of the off-axis scatter of the spatial coordinates.
    expr_noise : float, default 0.05
        Standard deviation of the pure-noise genes.
    marker_noise : float, default 0.05
        Standard deviation of the noise on the two marker genes; ``0.0`` makes expression
        exactly monotone in ``t``.
    n_comps : int, default 5
        Principal components for :func:`scanpy.pp.pca`.
    n_neighbors : int, default 12
        Neighbors for :func:`scanpy.pp.neighbors`.

    Returns
    -------
    adata : anndata.AnnData
        ``n`` cells with ``obs["clusters"]``, ``obsm["spatial"]``, ``obsm["X_pca"]``
        and an expression kNN graph under the default ``"neighbors"`` key.
    adata_dist : anndata.AnnData
        Dense ``(n, n)`` pairwise spatial distance matrix, in the shape
        :func:`deltatrace.pp.graphs` expects for its ``adata_dist`` argument, with both
        axes labeled by ``adata.obs_names``.
    """
    import scanpy as sc

    if n < 3:
        raise ValueError(f"line_toy(): need at least 3 cells to span source/mid/sink, got {n}.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)

    coords = np.column_stack([t * spatial_scale, rng.normal(0, spatial_jitter, n)])

    markers = [t, 1.0 - t]
    if marker_noise:
        markers = [m + rng.normal(0, marker_noise, n) for m in markers]
    X = np.column_stack(
        markers + [rng.normal(0, expr_noise, n) for _ in range(n_noise_genes)]
    ).astype(np.float32)

    labels = np.where(t < 1 / 3, "source", np.where(t < 2 / 3, "mid", "sink"))
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"clusters": labels}, index=[f"c{i}" for i in range(n)]),
    )
    adata.obsm["spatial"] = coords

    sc.pp.pca(adata, n_comps=min(n_comps, n - 1))
    sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, n - 1), use_rep="X_pca")

    D = cdist(coords, coords)
    np.fill_diagonal(D, 0.0)
    # Label both axes with the cell names: pp.integrate matches the pair by obs_names.
    adata_dist = ad.AnnData(
        X=D.astype(np.float64),
        obs=pd.DataFrame(index=adata.obs_names.copy()),
        var=pd.DataFrame(index=adata.obs_names.copy()),
    )

    return adata, adata_dist

