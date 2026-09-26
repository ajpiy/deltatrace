"""Preprocessing namespace (``deltatrace.pp``): builds the inputs the tools consume.

Holds :func:`wasserstein` (cell-cell spatial distance from a caller-supplied cell-to-spot
mapping), :func:`integrate` (one expression graph constrained by that distance) and
:func:`graphs` (every graph mode merged into ONE object, so a baseline and a spatially
constrained graph can be compared without losing provenance).
"""

from ._preprocessing import graphs, integrate, wasserstein

__all__ = ["wasserstein", "integrate", "graphs"]
