"""Preprocessing namespace (``deltatrace.pp``): builds the inputs the tools consume.

Holds :func:`wasserstein` (cell-cell spatial distance from a caller-supplied cell-to-spot
mapping) and :func:`integrate` (integrating spatial distance constraints with single-cell
expression graphs).
"""

from ._preprocessing import integrate, wasserstein

__all__ = ["wasserstein", "integrate"]
