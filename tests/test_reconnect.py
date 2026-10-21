"""Off-LCC pseudotime repair.

The QROT dual is a potential on the graph, so it is pinned only *within* a connected
component: on any other component it is free up to an additive constant. Cells outside the
largest connected component therefore carry a meaningless value and must be reassigned from
the LCC — and, just as importantly, must never be used as donors for that reassignment.
(``reconnect=False`` takes the other route, the LCC median; see ``test_off_lcc_median.py``.)

These pin both halves. The trap they guard is a mask defined as ``degree == 0``, which
catches only fully isolated cells and lets a two-cell island keep (and hand out) its
unpinned dual.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from deltatrace._transport import _reconnect_pseudotime
from deltatrace._utils import _off_lcc_mask

N = 8
LCC = [0, 1, 2, 3, 4, 5]        # a path
ISLAND = [6, 7]                 # a separate 2-cell component: degree 1 each, NOT isolated


def _sym(pairs, n=N, w=1.0):
    rows = [p for a, b in pairs for p in (a, b)]
    cols = [p for a, b in pairs for p in (b, a)]
    return sp.csr_matrix(([w] * len(rows), (rows, cols)), shape=(n, n))


@pytest.fixture
def basis_and_clust():
    """basis: intersection graph with an LCC + a detached island.
    clust: expression graph that DOES span both, so reconnection has a route."""
    names = [f"c{i}" for i in range(N)]

    intr = _sym([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),      # LCC
                 (6, 7)])                                      # island
    basis = ad.AnnData(X=np.zeros((N, 2), np.float32), obs=pd.DataFrame(index=names))
    basis.obsp["neighbors_intr_distances"] = intr
    basis.obsp["neighbors_intr_connectivities"] = intr
    basis.uns["neighbors_intr"] = {"distances_key": "neighbors_intr_distances",
                                   "connectivities_key": "neighbors_intr_connectivities"}

    expr = _sym([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    basis.obsp["neighbors_expr_distances"] = expr
    basis.uns["neighbors_expr"] = {"distances_key": "neighbors_expr_distances",
                                   "connectivities_key": "neighbors_expr_distances"}
    clust = ad.AnnData(X=np.zeros((N, 2), np.float32), obs=pd.DataFrame(index=names))
    clust.obsp["neighbors_expr_distances"] = expr
    clust.uns["neighbors_expr"] = {"distances_key": "neighbors_expr_distances",
                                   "connectivities_key": "neighbors_expr_distances"}
    return basis, clust


def test_off_lcc_mask_flags_islands_not_just_isolated_cells(basis_and_clust):
    basis, _ = basis_and_clust
    mask = _off_lcc_mask(basis, "intr")
    assert mask is not None
    assert list(np.where(mask)[0]) == ISLAND          # the island, though its degree is 1
    assert not mask[LCC].any()


def test_off_lcc_mask_returns_none_without_a_graph():
    """Callers fall back to the obs bookkeeping column when connectivity is unknowable."""
    bare = ad.AnnData(X=np.zeros((3, 2), np.float32),
                      obs=pd.DataFrame(index=["a", "b", "c"]))
    assert _off_lcc_mask(bare, None) is None


def test_island_cells_are_reassigned_from_the_lcc(basis_and_clust):
    """An island's own dual is arbitrary; after reconnection it must equal some LCC value."""
    basis, _ = basis_and_clust
    lcc_pt = {f"c{i}": i / 5.0 for i in LCC}          # 0.0 .. 1.0 along the path
    arbitrary = 0.8179                                # the unpinned island dual
    pt = dict(lcc_pt, **{f"c{i}": arbitrary for i in ISLAND})

    out = _reconnect_pseudotime(basis, pt, graph="intr",
                                reconnect_graph="expr")

    for i in ISLAND:
        assert out[f"c{i}"] != pytest.approx(arbitrary), "island kept its unpinned dual"
        assert out[f"c{i}"] in set(lcc_pt.values()), "island value did not come from the LCC"
    # nearest LCC cell to both 6 and 7 along the expression path is c5 (pt = 1.0)
    assert out["c6"] == pytest.approx(1.0)
    assert out["c7"] == pytest.approx(1.0)


def test_lcc_pseudotime_is_untouched(basis_and_clust):
    basis, _ = basis_and_clust
    lcc_pt = {f"c{i}": i / 5.0 for i in LCC}
    pt = dict(lcc_pt, **{f"c{i}": 0.8179 for i in ISLAND})
    out = _reconnect_pseudotime(basis, pt, graph="intr",
                                reconnect_graph="expr")
    for k, v in lcc_pt.items():
        assert out[k] == pytest.approx(v)


def test_an_island_can_never_be_a_donor(basis_and_clust):
    """The donor pool is the complement of the repair mask, so widening the mask to
    off-LCC also removes islands as donors. Give the island a value no LCC cell has: if it
    ever leaks into another cell, that cell would carry it."""
    basis, _ = basis_and_clust
    sentinel = -12345.0
    lcc_pt = {f"c{i}": i / 5.0 for i in LCC}
    pt = dict(lcc_pt, **{f"c{i}": sentinel for i in ISLAND})

    out = _reconnect_pseudotime(basis, pt, graph="intr",
                                reconnect_graph="expr")

    assert sentinel not in out.values(), "an unpinned island dual survived or propagated"


def test_reconnect_assigns_every_cell(basis_and_clust):
    basis, _ = basis_and_clust
    pt = dict({f"c{i}": i / 5.0 for i in LCC}, **{f"c{i}": 0.5 for i in ISLAND})
    out = _reconnect_pseudotime(basis, pt, graph="intr",
                                reconnect_graph="expr")
    assert set(out) == {f"c{i}" for i in range(N)}
    assert all(np.isfinite(v) for v in out.values())


def test_qrot_docstring_says_what_reconnect_false_actually_writes():
    """reconnect=False sets the off-LCC cells to the LCC median, and never left them NaN.

    Callers read this docstring to know what they get, so it has to describe the
    placeholder it writes -- and say that it IS a placeholder, not an inferred position."""
    import deltatrace as dt

    doc = dt.tl.qrot.__doc__
    assert "stay NaN" not in doc
    assert "WARNING" in doc
    assert "median" in doc                                # the placeholder it writes
    assert "NOT" in doc and "inferred position" in doc     # ...and that it is not a result
    # NOT obs["disconnected_<graph>"]: that column is written as degree == 0 and only for
    # intersection graphs, so it misses multi-cell islands.
    # endpoints' disconnected_info is the set that actually matches _off_lcc_mask.
    assert "disconnected_info" in doc                     # what to read instead


def test_off_lcc_cells_are_not_left_nan_when_reconnect_is_off(basis_and_clust):
    """Run the real reconnect=False path and pin what it produces.

    The island cells come back finite and on the LCC median, and nothing in the written
    column separates them from a genuine intermediate cell -- only the off-LCC bookkeeping
    does. ``graph["disconnected_info"]`` is the authoritative set; ``obs["disconnected_"]``
    is only the ``degree == 0`` subset of it, so it is deliberately NOT asserted here.
    """
    import deltatrace as dt

    basis, _ = basis_and_clust
    basis.obs["clusters"] = pd.Categorical(
        ["source", "source", "mid", "mid", "sink", "sink", "mid", "mid"]
    )
    graph = dt.tl.endpoints(basis, ["source"], ["sink"], cluster_key="clusters",
                            graph="intr", p_end=0.2, verbose=0)
    dt.tl.qrot(graph, adata=basis, graph="intr", reconnect=False, verbose=0)

    pt = basis.obs["pseudotime"].to_numpy()
    island, lcc = pt[ISLAND], pt[LCC]
    assert np.isfinite(island).all(), "reconnect=False must NOT leave them NaN"
    # before it they had no potential (no source or sink in their piece): NaN
    assert np.isnan(basis.obs["pseudotime_raw"].to_numpy()[ISLAND]).all()
    # the explicit placeholder
    assert island == pytest.approx([float(np.median(lcc))] * len(ISLAND))
    # ...and the only thing that identifies them is the off-LCC bookkeeping, which the
    # caller has to consult and mask by; nothing in the written column does it for them.
    assert graph["disconnected_info"]["indices"] == ISLAND
    assert list(np.where(_off_lcc_mask(basis, "intr"))[0]) == ISLAND


def test_reconnect_true_moves_those_same_cells_onto_an_lcc_value(basis_and_clust):
    """The other side of the same coin: with reconnect on they take an LCC cell's value."""
    import deltatrace as dt

    basis, _ = basis_and_clust
    basis.obs["clusters"] = pd.Categorical(
        ["source", "source", "mid", "mid", "sink", "sink", "mid", "mid"]
    )
    graph = dt.tl.endpoints(basis, ["source"], ["sink"], cluster_key="clusters",
                            graph="intr", p_end=0.2, verbose=0)
    dt.tl.qrot(graph, adata=basis, graph="intr", reconnect=True,
               reconnect_graph="expr", verbose=0)

    pt = basis.obs["pseudotime"].to_numpy()
    assert set(np.round(pt[ISLAND], 12)) <= set(np.round(pt[LCC], 12))
