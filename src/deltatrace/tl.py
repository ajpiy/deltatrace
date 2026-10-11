"""Tools namespace (``deltatrace.tl``): the trajectory pipeline itself.

Holds :func:`endpoints`,
:func:`qrot`, :func:`compute_velocity_grid`, :func:`lineages`, :func:`terminality`,
:func:`widest_path` (an explicit route between two cells, on demand) and
:func:`cluster_trajectory` (the flow's own cluster-level graph); :func:`trajectory` runs
the first four in one call, and every stage stays public rather than reachable only
through that wrapper. :class:`CellGraph` is the cell graph :func:`endpoints` hands to
:func:`qrot`. :func:`qrot_graph` runs the same solver on any graph in SciPy's csgraph form.
"""

from ._endpoints import endpoints
from ._lineages import cluster_trajectory, lineages, terminality, widest_path
from ._transport import compute_velocity_grid, qrot, qrot_graph, trajectory
from ._utils import CellGraph

__all__ = [
    "trajectory",
    "endpoints",
    "CellGraph",
    "qrot",
    "qrot_graph",
    "compute_velocity_grid",
    "lineages",
    "terminality",
    "widest_path",
    "cluster_trajectory",
]
