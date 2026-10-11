"""DeltaTrace: spatially-regularized trajectory inference for scRNA-seq and spatial data.

Cells are ordered by quadratically regularized optimal transport (QROT) run on a
cell-cell graph that physical distance has already constrained, so a cell only advances
to a neighbor it is close to both in expression and in the tissue. One conic solve
returns the dual potential, taken directly as pseudotime, and the primal directed flow,
from which lineages are read off an absorbing Markov chain.

Main components
---------------
- pp: preprocessing of spatial cell-cell distances and integrated expression/spatial graphs
- tl: tools for endpoints, QROT pseudotime and flow, velocity grid, lineages, the
  flow-native cluster trajectory
- pl: plotting for pseudotime with velocity, the velocity layer alone, endpoints,
  lineages, the cluster trajectory
- datasets: a synthetic crypt, the example system for the quickstart and tutorials

Quick Start
-----------
>>> import deltatrace as dt
>>>
>>> # crypt() hands back a pair: expression object + tissue distance matrix
>>> adata, adata_dist = dt.datasets.crypt()
>>>
>>> # expression + intersection graphs, merged into one object
>>> ad_intg = dt.pp.graphs(adata, adata_dist)
building the 'expr' graph ... finished (...)
building the 'intr' graph ... finished (...)
done.
>>>
>>> # endpoints -> qrot -> velocity -> lineages
>>> _ = dt.tl.trajectory(ad_intg, ["progenitor"], ["fate_A", "fate_B"], basis="spatial")
sampling source and sink endpoints ... finished (...)
solving quadratically regularized transport ... finished (...)
building the velocity grid ... finished (...)
assigning lineages by absorbing chain ... finished (...)
done.
>>>
>>> # cells colored by pseudotime, with velocity arrows
>>> _ = dt.pl.pseudotime(ad_intg, basis="spatial")

Every stage reports itself like that; pass ``verbose=0`` to silence it, or ``verbose=2``
for per-step detail (edge counts, how many cells fell off the graph).

On real scRNA-seq the distance matrix comes from :func:`deltatrace.pp.wasserstein`
instead, given a cell-to-spot mapping onto a paired spatial section.

References
----------
Essid M, Solomon J. Quadratically regularized optimal transport on graphs.
SIAM J. Sci. Comput. 40(4):A1961-A1986 (2018). doi:10.1137/17M1132665

The DeltaTrace manuscript is in preparation; until then, cite the software via
CITATION.cff.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from . import datasets, pl, pp, tl

__all__ = ["pp", "tl", "pl", "datasets", "__version__"]

try:
    # single source of truth: the version declared in pyproject.toml
    __version__ = _pkg_version("deltatrace")
except PackageNotFoundError:          # running from a source tree, not installed
    __version__ = "0.0.0+unknown"
