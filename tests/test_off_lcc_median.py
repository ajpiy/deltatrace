"""Off-LCC pseudotime under ``reconnect=False``: the LCC median, deliberately.

A cell outside the largest connected component carries no source or sink demand, so its
dual potential is whatever the solver's least-norm solution returned -- near 0, which the
min-max scaling then lands near the middle of the range. That is a plausible-looking
mid-trajectory value arrived at BY ACCIDENT. With ``reconnect=False`` those cells are now
all set to the median of the LCC cells' pseudotime instead: still a placeholder, but an
explicit and reproducible one. ``reconnect=True`` keeps reconnecting along geodesics and
must not shift.
"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import deltatrace as dt
from deltatrace._transport import _median_off_lcc_pseudotime
from deltatrace._utils import _off_lcc_mask

N = 8
LCC = [0, 1, 2, 3, 4, 5]        # a path
ISLAND = [6, 7]                 # a separate 2-cell component: degree 1 each, NOT isolated
CLUSTERS = ["source", "source", "mid", "mid", "sink", "sink", "mid", "mid"]


def _sym(pairs, n=N, w=1.0):
    rows = [p for a, b in pairs for p in (a, b)]
    cols = [p for a, b in pairs for p in (b, a)]
    return sp.csr_matrix(([w] * len(rows), (rows, cols)), shape=(n, n))


def _basis(intr_pairs):
    """An object carrying just the integrated graph the LCC is read from."""
    names = [f"c{i}" for i in range(N)]
    obj = ad.AnnData(X=np.zeros((N, 2), np.float32), obs=pd.DataFrame(index=names))
    graph = _sym(intr_pairs)
    obj.obsp["neighbors_intr_distances"] = graph
    obj.obsp["neighbors_intr_connectivities"] = graph
    obj.uns["neighbors_intr"] = {"distances_key": "neighbors_intr_distances",
                                 "connectivities_key": "neighbors_intr_connectivities"}
    obj.obs["clusters"] = pd.Categorical(CLUSTERS)
    return obj


@pytest.fixture
def split_basis():
    """intr graph = a 6-cell LCC path + a detached 2-cell island."""
    return _basis([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7)])


@pytest.fixture
def whole_basis():
    """intr graph = one path over every cell, so nothing is off the LCC."""
    return _basis([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])


@pytest.fixture
def clust():
    """Expression graph that spans the island too, so reconnection has a route."""
    names = [f"c{i}" for i in range(N)]
    obj = ad.AnnData(X=np.zeros((N, 2), np.float32), obs=pd.DataFrame(index=names))
    obj.obsp["neighbors_expr_distances"] = _sym(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    )
    obj.uns["neighbors_expr"] = {"distances_key": "neighbors_expr_distances",
                                 "connectivities_key": "neighbors_expr_distances"}
    return obj


def _solve(basis, adata_clust=None, **kw):
    graph = dt.tl.endpoints(basis, basis, ["source"], ["sink"], cluster_key="clusters",
                            graph="intr", p_end=0.2, verbose=0)
    dt.tl.qrot(graph, adata_basis=basis, adata_clust=adata_clust, graph="intr",
               reconnect_graph="expr", verbose=0, **kw)
    return graph


def test_off_lcc_cells_get_the_lcc_median_when_reconnect_is_off(split_basis):
    """The whole point: one explicit placeholder, equal to the median of the LCC values."""
    graph = _solve(split_basis, reconnect=False)

    pt = split_basis.obs["pseudotime"].to_numpy()
    island, lcc = pt[ISLAND], pt[LCC]
    expected = float(np.median(lcc))

    assert np.isfinite(island).all()
    assert island == pytest.approx([expected] * len(ISLAND))
    # the cells the median was written to are exactly the off-LCC ones
    assert list(np.where(_off_lcc_mask(split_basis, "intr"))[0]) == ISLAND
    assert graph["disconnected_info"]["indices"] == ISLAND
    # not the arbitrary dual any more: pseudotime_raw still holds that, and it differs
    raw_island = split_basis.obs["pseudotime_raw"].to_numpy()[ISLAND]
    assert np.isfinite(raw_island).all()
    assert raw_island[0] != pytest.approx(expected)


def test_reconnect_true_still_takes_an_lcc_cells_value_not_the_median(split_basis, clust):
    """reconnect=True is untouched: geodesic donor, which here is NOT the LCC median."""
    reconnected = _basis([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7)])
    _solve(reconnected, adata_clust=clust, reconnect=True)
    _solve(split_basis, reconnect=False)

    pt_on = reconnected.obs["pseudotime"].to_numpy()
    pt_off = split_basis.obs["pseudotime"].to_numpy()

    assert pt_on[LCC] == pytest.approx(pt_off[LCC]), "the LCC solve must not move"
    # donor values come from the LCC; the nearest LCC cell to both 6 and 7 is c5
    assert set(np.round(pt_on[ISLAND], 12)) <= set(np.round(pt_on[LCC], 12))
    assert pt_on[ISLAND] == pytest.approx([pt_on[5]] * len(ISLAND))
    # and that is a different answer from the reconnect=False placeholder
    median = float(np.median(pt_off[LCC]))
    assert pt_on[ISLAND][0] != pytest.approx(median)
    assert pt_off[ISLAND] == pytest.approx([median] * len(ISLAND))


def test_no_off_lcc_cells_is_a_no_op(whole_basis):
    """Nothing off the LCC -> the mapping comes back untouched, not flattened to a median."""
    pt = {f"c{i}": i / (N - 1) for i in range(N)}
    out = _median_off_lcc_pseudotime(whole_basis, pt, graph="intr")

    assert out == pt
    assert out is not pt, "the helper must not alias the caller's dict"
    assert not _off_lcc_mask(whole_basis, "intr").any()


def test_no_off_lcc_cells_is_a_no_op_through_qrot(whole_basis):
    """The same through the real reconnect=False path: pseudotime == the scaled potentials."""
    _solve(whole_basis, reconnect=False)

    pt = whole_basis.obs["pseudotime"].to_numpy()
    raw = whole_basis.obs["pseudotime_raw"].to_numpy()
    assert pt == pytest.approx(raw, abs=0.0), "no off-LCC cell, so nothing may be rewritten"
    assert len(set(np.round(pt, 12))) > 1, "a median fill would have flattened these"


def test_an_empty_lcc_neither_crashes_nor_warns(split_basis):
    """No LCC cell carries a pseudotime: return as-is, and never reach an all-NaN median."""
    only_island = {f"c{i}": 0.4 for i in ISLAND}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # any RuntimeWarning fails the test
        out = _median_off_lcc_pseudotime(split_basis, only_island, graph="intr")
    assert out == only_island


def test_a_single_lcc_cell_gives_its_own_value_as_the_median(split_basis):
    """Median of one value is that value, so the island lands exactly on it."""
    pt = dict({"c3": 0.31}, **{f"c{i}": 0.99 for i in ISLAND})
    out = _median_off_lcc_pseudotime(split_basis, pt, graph="intr")
    assert out["c3"] == pytest.approx(0.31)
    for i in ISLAND:
        assert out[f"c{i}"] == pytest.approx(0.31)


def test_a_missing_graph_leaves_pseudotime_alone():
    """Without a graph there is no LCC to read, so the helper cannot and does not guess."""
    bare = ad.AnnData(X=np.zeros((3, 2), np.float32),
                      obs=pd.DataFrame(index=["a", "b", "c"]))
    pt = {"a": 0.0, "b": 0.7, "c": 1.0}
    assert _off_lcc_mask(bare, None) is None
    assert _median_off_lcc_pseudotime(bare, pt) == pt
    assert _median_off_lcc_pseudotime(None, pt) == pt


def test_the_median_is_logged_with_its_count_and_value(split_basis, capsys):
    """_log.detail is verbose=2 only, and must say how many cells were set and to what."""
    graph = dt.tl.endpoints(split_basis, split_basis, ["source"], ["sink"],
                            cluster_key="clusters", graph="intr", p_end=0.2, verbose=0)
    dt.tl.qrot(graph, adata_basis=split_basis, graph="intr", reconnect=False, verbose=2)
    out = capsys.readouterr().out

    median = float(np.median(split_basis.obs["pseudotime"].to_numpy()[LCC]))
    assert f"set {len(ISLAND)} off-LCC cells to the LCC median" in out
    assert f"{median:.6f}" in out


def test_the_obs_disconnected_column_does_not_identify_every_cell_that_was_set(split_basis):
    """Guards the docstring's route: ``obs["disconnected_"]`` is only a SUBSET of the set.

    ``obs["disconnected_<graph>"]`` is written by :func:`pp.integrate` as ``degree == 0``, so a
    multi-cell island off the LCC is absent from it while still receiving the median. This
    pins why ``qrot`` points at ``endpoints``' ``"disconnected_info"`` instead.
    """
    graph = dt.tl.endpoints(split_basis, split_basis, ["source"], ["sink"],
                            cluster_key="clusters", graph="intr", p_end=0.2, verbose=0)
    dt.tl.qrot(graph, adata_basis=split_basis, graph="intr", reconnect=False, verbose=0)

    off = _off_lcc_mask(split_basis, "intr")
    assert list(np.where(off)[0]) == ISLAND

    # the authoritative route agrees with the cells that were actually rewritten
    assert graph["disconnected_info"]["indices"] == ISLAND

    # the obs column, being degree==0, catches NONE of this 2-cell island
    degree = np.diff(split_basis.obsp["neighbors_intr_distances"].tocsr().indptr)
    assert (degree[ISLAND] > 0).all(), "the island is not isolated, so degree==0 misses it"
    assert not np.array_equal(degree == 0, off)
