"""Plotting namespace (``deltatrace.pl``): visualizations of DeltaTrace results.

Holds :func:`pseudotime` (cells colored by pseudotime, with velocity arrows),
:func:`velocity` (the arrows alone, on any axes -- e.g. one from ``scanpy.pl.embedding``),
:func:`endpoints` (the sampled source and sink cells), :func:`lineages`
(absorbing-chain fates) and :func:`cluster_trajectory` (the flow's own cluster graph).

Every quantity these functions color by is also a plain ``obs`` column (``pseudotime``,
``lineage``, ``lineage_confidence``, ``fate_<lineage>``, ``endpoint_<graph>``), so
``scanpy.pl.embedding(adata, basis=..., color=...)`` draws the same data.
"""

from ._plotting import (
    cluster_trajectory,
    endpoints,
    lineages,
    pseudotime,
    velocity,
)

__all__ = ["pseudotime", "velocity", "endpoints", "lineages",
           "cluster_trajectory"]
