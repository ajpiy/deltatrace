"""Tools namespace (``deltatrace.tl``): the trajectory pipeline itself.

Holds :func:`endpoints`,
:func:`qrot`, :func:`compute_velocity_grid`, :func:`lineages`, :func:`terminality`,
:func:`widest_path` (an explicit route between two cells, on demand) and
:func:`cluster_trajectory` (the flow's own cluster-level graph); :func:`trajectory` runs
the first four in one call, and every stage stays public rather than reachable only
through that wrapper. :class:`CellGraph` is the cell graph :func:`endpoints` hands to
:func:`qrot`.
"""

from ._endpoints import endpoints
from ._lineages import cluster_trajectory, lineages, terminality, widest_path
from ._transport import compute_velocity_grid, qrot, trajectory
from ._utils import CellGraph

__all__ = [
    "trajectory",
    "endpoints",
    "CellGraph",
    "qrot",
    "compute_velocity_grid",
    "lineages",
    "terminality",
    "widest_path",
    "cluster_trajectory",
]
