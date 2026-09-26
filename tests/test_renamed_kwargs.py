"""The 0.1.0 names: ``graph=`` picks the stored graph, ``reconnect_graph=`` the one off-LCC
cells reconnect through, and ``pp.integrate(key_added=)`` names what it stores.

The earlier names (``tag=``, ``reconnect_key="neighbors_<graph>"``, ``integrate(tag=)``)
still work with a FutureWarning, and must give byte-identical results: pipelines written
against them have to keep reproducing.
"""

import inspect
import warnings

import matplotlib
import numpy as np
import pytest
from _line_toy import line_toy

import deltatrace as dt
from deltatrace._utils import _off_lcc_mask

matplotlib.use("Agg")

COMMON = dict(cluster_key="clusters", p_end=0.10, seed=3, verbose=0)


@pytest.fixture(scope="module")
def graphed():
    return dt.pp.graphs(*line_toy(n=120), alpha=0.5, p_keep=0.1, verbose=0)


def test_public_signatures_carry_only_the_new_names():
    for fn in (dt.tl.endpoints, dt.tl.qrot, dt.tl.trajectory, dt.pl.endpoints):
        params = inspect.signature(fn).parameters
        assert "graph" in params and "tag" not in params, fn.__name__
    for fn in (dt.tl.qrot, dt.tl.trajectory):
        params = inspect.signature(fn).parameters
        assert "reconnect_graph" in params and "reconnect_key" not in params, fn.__name__
    params = inspect.signature(dt.pp.integrate).parameters
    assert "key_added" in params and "tag" not in params


def test_old_tag_on_endpoints_warns_and_matches(graphed):
    a, b = graphed.copy(), graphed.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        new = dt.tl.endpoints(a, a, ["source"], ["sink"], graph="intr", **COMMON)
    with pytest.warns(FutureWarning, match=r"tag= is deprecated.*graph='intr'") as rec:
        old = dt.tl.endpoints(b, b, ["source"], ["sink"], tag="intr", **COMMON)
    assert rec[0].filename == __file__                 # points at the caller's line
    assert new["source_cells"] == old["source_cells"]
    assert new["sink_cells"] == old["sink_cells"]
    np.testing.assert_array_equal(new["f"], old["f"])
    assert (a.obs["endpoint_intr"] == b.obs["endpoint_intr"]).all()


def test_old_names_on_qrot_warn_and_match(graphed):
    a, b = graphed.copy(), graphed.copy()
    ga = dt.tl.endpoints(a, a, ["source"], ["sink"], graph="intr", **COMMON)
    gb = dt.tl.endpoints(b, b, ["source"], ["sink"], graph="intr", **COMMON)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        dt.tl.qrot(ga, adata_basis=a, adata_clust=a, graph="intr", reconnect_graph="expr",
                   verbose=0)
    with pytest.warns(FutureWarning) as rec:
        dt.tl.qrot(gb, adata_basis=b, adata_clust=b, tag="intr",
                   reconnect_key="neighbors_expr", verbose=0)
    msgs = " ".join(str(w.message) for w in rec)
    assert "graph='intr'" in msgs and "reconnect_graph='expr'" in msgs
    np.testing.assert_array_equal(a.obs["pseudotime"], b.obs["pseudotime"])
    np.testing.assert_array_equal(a.uns["dt_flow"]["J"], b.uns["dt_flow"]["J"])


def test_old_names_on_trajectory_warn_and_match(graphed):
    a, b = graphed.copy(), graphed.copy()
    kw = dict(basis="spatial", **COMMON)
    dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", reconnect_graph="expr", **kw)
    with pytest.warns(FutureWarning):
        dt.tl.trajectory(b, ["source"], ["sink"], tag="intr",
                         reconnect_key="neighbors_expr", **kw)
    for col in ("pseudotime", "lineage", "lineage_confidence"):
        np.testing.assert_array_equal(a.obs[col], b.obs[col])


def test_old_tag_on_integrate_names_the_graph():
    adata, dist = line_toy(n=80)
    new = dt.pp.integrate(adata, dist, key_added="mine", verbose=0)
    with pytest.warns(FutureWarning, match="key_added='mine'"):
        old = dt.pp.integrate(adata, dist, tag="mine", verbose=0)
    for out in (new, old):
        assert "neighbors_mine_distances" in out.obsp
        assert out.uns["neighbors_mine"]["params"]["method"] == "mine"
    assert (new.obsp["neighbors_mine_distances"] != old.obsp["neighbors_mine_distances"]).nnz == 0
    # and a custom name is what graph= then selects
    dt.tl.endpoints(new, new, ["source"], ["sink"], graph="mine", **COMMON)
    assert "endpoint_mine" in new.obs


def test_old_tag_on_pl_endpoints(graphed):
    a = graphed.copy()
    gr = dt.tl.endpoints(a, a, ["source"], ["sink"], graph="intr", **COMMON)
    with pytest.warns(FutureWarning):
        _, ax = dt.pl.endpoints(gr, a, basis="spatial", tag="intr", show=False)
    assert ax.get_title() == "Endpoints: intr"


def _ep(a):
    return dt.tl.endpoints(a, a, ["source"], ["sink"], graph="intr", **COMMON)


# name -> (number of old names passed, the call)
OLD_NAME_CALLS = {
    "tl.endpoints": (1, lambda a: dt.tl.endpoints(a, a, ["source"], ["sink"], tag="intr",
                                                  **COMMON)),
    "tl.qrot": (2, lambda a: dt.tl.qrot(_ep(a), adata_basis=a, adata_clust=a, tag="intr",
                                        reconnect_key="neighbors_expr", verbose=0)),
    "tl.trajectory": (2, lambda a: dt.tl.trajectory(a, ["source"], ["sink"], tag="intr",
                                                    reconnect_key="neighbors_expr",
                                                    basis="spatial", **COMMON)),
    "pl.endpoints": (1, lambda a: dt.pl.endpoints(_ep(a), a, basis="spatial", tag="intr",
                                                  show=False)),
    "pp.integrate": (1, lambda a: dt.pp.integrate(*line_toy(n=60), tag="mine",
                                                  verbose=0)),
}


@pytest.mark.parametrize("name", list(OLD_NAME_CALLS))
def test_each_old_name_warns_once_at_the_callers_line(graphed, name):
    """One FutureWarning per old name, each attributed to the caller: an internal call still
    using an old name would add a warning pointing inside the package."""
    n_old, call = OLD_NAME_CALLS[name]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        call(graphed.copy())
    ours = [w for w in rec if issubclass(w.category, FutureWarning)
            and "is deprecated and will be removed; use" in str(w.message)]
    assert len(ours) == n_old
    assert all(w.filename == __file__ for w in ours)


@pytest.mark.parametrize("call", [
    lambda a: dt.tl.endpoints(a, a, ["source"], ["sink"], graph="intr", tag="intr", **COMMON),
    lambda a: dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", basis="spatial",
                               reconnect_graph="expr", reconnect_key="neighbors_expr",
                               **COMMON),
    lambda a: dt.pp.integrate(*line_toy(n=60), key_added="x", tag="x", verbose=0),
], ids=["tag+graph", "reconnect_key+reconnect_graph", "integrate tag+key_added"])
def test_both_spellings_at_once_is_an_error(graphed, call):
    a = graphed.copy()
    with pytest.raises(TypeError, match="got both"):
        call(a)
    assert "endpoint_intr" not in a.obs               # refused before any work


# --------------------------------------------------------------------------------------
# reconnect_graph: the route off-LCC cells are reconnected through
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def islanded():
    """Three cells off the intersection graph's LCC, so reconnection really runs, and a
    second route to them: a scanpy graph stored with key_added="nbr_space"."""
    import scanpy as sc
    g = dt.pp.graphs(*line_toy(n=150), alpha=0.5, p_keep=0.08, verbose=0)
    sc.pp.neighbors(g, n_neighbors=5, use_rep="spatial", key_added="nbr_space")
    return g


def _reconnected(obj, **route):
    x = obj.copy()
    gr = dt.tl.endpoints(x, x, ["source"], ["sink"], graph="intr", seed=1, verbose=0)
    dt.tl.qrot(gr, adata_basis=x, adata_clust=x, graph="intr", verbose=0, **route)
    return x.obs["pseudotime"].to_numpy()


def test_reconnect_graph_picks_the_route_and_the_old_key_matches(islanded):
    off = _off_lcc_mask(islanded, "intr")
    assert off.sum() > 0                               # premise: reconnection runs
    via_expr = _reconnected(islanded)                  # default: reconnect_graph="expr"
    via_space = _reconnected(islanded, reconnect_graph="nbr_space")
    assert not np.array_equal(via_expr[off], via_space[off])   # the choice reaches the result
    np.testing.assert_array_equal(via_expr[~off], via_space[~off])
    with pytest.warns(FutureWarning, match="reconnect_graph='expr'"):
        old_expr = _reconnected(islanded, reconnect_key="neighbors_expr")
    # a scanpy key_added graph, reachable by reconnect_key before 0.1.0 and still now
    with pytest.warns(FutureWarning, match="reconnect_graph='nbr_space'"):
        old_space = _reconnected(islanded, reconnect_key="nbr_space")
    np.testing.assert_array_equal(old_expr, via_expr)
    np.testing.assert_array_equal(old_space, via_space)


def test_a_missing_reconnect_graph_is_an_error(islanded):
    with pytest.raises(ValueError, match="No graph 'nope'"):
        _reconnected(islanded, reconnect_graph="nope")


@pytest.mark.parametrize("bad", [None, "neighbors_"])
def test_reconnect_key_must_name_a_graph(islanded, bad):
    with pytest.raises(ValueError, match="reconnect_graph"):
        _reconnected(islanded, reconnect_key=bad)


def test_multiple_graphs_error_names_graph(graphed):
    a = graphed.copy()
    with pytest.raises(ValueError, match="Pass graph= explicitly"):
        dt.tl.endpoints(a, a, ["source"], ["sink"], **COMMON)


def test_the_old_result_key_still_reads_the_endpoints(graphed):
    a = graphed.copy()
    res = dt.tl.trajectory(a, ["source"], ["sink"], graph="intr", basis="spatial", **COMMON)
    assert set(res) == {"endpoints", "lineages"}
    with pytest.warns(FutureWarning, match=r'\["endpoints"\]') as rec:
        old = res["graph"]
    assert old is res["endpoints"]
    assert rec[0].filename == __file__
    with pytest.raises(KeyError):
        res["nope"]
