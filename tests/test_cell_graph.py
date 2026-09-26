"""endpoints' cell graph (CellGraph) against the networkx.DiGraph it replaced.

endpoints used to build a networkx.DiGraph and read everything back out of it; it now
keeps the cost matrix as scipy arrays. The QROT solve is deterministic for a fixed edge
order but not invariant to it, so the edge order, the costs and every consumer of the old
graph must line up exactly with what networkx gave: compared with ==, never allclose.
"""

import copy
import pickle

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from _line_toy import line_toy

import deltatrace as dt
from deltatrace._utils import CellGraph, _build_cell_graph, _build_digraph


@pytest.fixture(scope="module")
def toy():
    return dt.pp.graphs(*line_toy(n=200), alpha=0.5, verbose=0)


def _edges_with_data(G):
    return [(u, v, dict(d)) for u, v, d in G.edges(data=True)]


def test_edges_and_costs_follow_the_networkx_edge_order(toy):
    cg = _build_cell_graph(toy, graph="intr")
    G = _build_digraph(toy, graph="intr")
    assert (cg.n, cg.m) == (G.number_of_nodes(), G.number_of_edges())
    assert cg.edge_array.dtype == np.int64 and cg.edge_array.shape == (cg.m, 2)
    assert cg.edge_array.tolist() == [list(e) for e in G.edges()]
    expect = np.array([d["cost"] for *_, d in G.edges(data=True)], dtype=np.float64)
    assert cg.cost_vec.dtype == np.float64
    assert cg.cost_vec.tobytes() == expect.tobytes()


def test_to_networkx_is_the_old_graph_with_its_metadata(toy):
    gr = dt.tl.endpoints(toy, toy, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                         p_end=0.05, verbose=0)
    cg = gr["G"]
    assert isinstance(cg, CellGraph)
    H = cg.to_networkx()
    G = _build_digraph(toy, graph="intr")
    assert isinstance(H, nx.DiGraph)
    assert list(H.nodes()) == list(G.nodes())
    # "cost" on every edge, "weight" only where the connectivities store one
    assert _edges_with_data(H) == _edges_with_data(G)
    assert list(H.graph) == list(cg.graph) == [
        "obs_names", "name_to_idx", "idx_to_name", "source_indices", "sink_indices",
        "source_names", "sink_names", "disconnected_info"]
    assert np.array_equal(H.graph["obs_names"], G.graph["obs_names"])
    for key in ("name_to_idx", "idx_to_name"):
        assert H.graph[key] == G.graph[key]
    assert H.graph["source_indices"] == gr["source_indices"]
    assert H.graph["sink_names"] == gr["sink_cells"]
    assert H.graph["disconnected_info"] == gr["disconnected_info"]


def _components_fixture():
    """Paths inside four components: P={0,7,8}, Q={1,2,3,4}, R={5,6,9,10}, S={11}.
    Q and R tie for largest; Q holds the lower cell, so Q must be the LCC."""
    n = 12
    pairs = [(0, 7), (7, 8), (1, 2), (2, 3), (3, 4), (5, 6), (6, 9), (9, 10)]
    rows = [u for u, v in pairs] + [v for u, v in pairs]
    cols = [v for u, v in pairs] + [u for u, v in pairs]
    D = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = D.copy()
    A.data[:] = 0.5
    clusters = ["mid", "source", "source", "sink", "sink"] + ["mid"] * 7
    obj = ad.AnnData(X=np.zeros((n, 1), np.float32),
                     obs=pd.DataFrame({"clusters": clusters}, index=[f"c{i}" for i in range(n)]))
    obj.obsp["neighbors_intr_distances"] = D
    obj.obsp["neighbors_intr_connectivities"] = A
    obj.uns["neighbors_intr"] = {"distances_key": "neighbors_intr_distances",
                                 "connectivities_key": "neighbors_intr_connectivities"}
    return obj


@pytest.mark.parametrize("mode", ["random", "centered", "directed"])
def test_largest_component_tie_goes_to_the_lowest_cell(mode):
    obj = _components_fixture()
    gr = dt.tl.endpoints(obj, obj, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                         p_end=0.5, sampling_mode=mode, verbose=0)
    info = gr["disconnected_info"]
    assert info["indices"] == [0, 5, 6, 7, 8, 9, 10, 11]
    assert info["names"] == [f"c{i}" for i in info["indices"]]
    assert info["component_sizes"] == [4, 4, 3, 1]
    # ...which is what the networkx version computed on the old graph
    comps = sorted(nx.connected_components(_build_digraph(obj, graph="intr").to_undirected()),
                   key=len, reverse=True)
    assert info["component_sizes"] == [len(c) for c in comps]
    assert info["indices"] == sorted(set(range(obj.n_obs)) - comps[0])
    assert set(gr["source_indices"]) | set(gr["sink_indices"]) <= comps[0]


def test_blocked_flows_bias_matches_a_per_edge_check(toy):
    # adata_clust misses some cells: they have no label, and match no blocked pair
    clust = toy[np.arange(toy.n_obs) % 10 != 3].copy()
    blocked = [("sink", "mid"), ("mid", "source"), ("source", "source"), ("nope", "sink")]
    gr = dt.tl.endpoints(toy, clust, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                         p_end=0.05, blocked_flows=blocked, verbose=0)
    name_to_idx = gr["G"].graph["name_to_idx"]
    lab = {name_to_idx[nm]: str(c) for nm, c in clust.obs["clusters"].items()}
    expect = np.array(
        [1.0 if (lab.get(u), lab.get(v)) in set(blocked) else d["cost"]
         for u, v, d in gr["G"].to_networkx().edges(data=True)], dtype=np.float64)
    assert (expect == 1.0).any() and (expect != 1.0).any()
    assert gr["cost_vec"].tobytes() == expect.tobytes()


def test_qrot_solves_the_same_problem_on_either_graph(toy):
    """qrot's CellGraph branch and its networkx branch must write identical results."""
    a_cg, a_nx = toy.copy(), toy.copy()
    gr = dt.tl.endpoints(a_cg, a_cg, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                         p_end=0.05, sampling_mode="directed", seed=0, verbose=0)
    gr_nx = dict(gr, G=gr["G"].to_networkx())
    dt.tl.qrot(gr, adata_basis=a_cg, adata_clust=a_cg, graph="intr", verbose=0)
    dt.tl.qrot(gr_nx, adata_basis=a_nx, adata_clust=a_nx, graph="intr", verbose=0)
    for key in ("pseudotime_raw", "pseudotime"):
        x, y = a_cg.obs[key].to_numpy(), a_nx.obs[key].to_numpy()
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), key
    for key in ("J", "edges"):
        x, y = a_cg.uns["dt_flow"][key], a_nx.uns["dt_flow"][key]
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), key


def test_to_networkx_is_frozen_when_endpoints_runs(toy):
    """endpoints used to build its DiGraph once, inside the call, so editing adata.obsp in
    place afterwards could not reach it. to_networkx() must not see such an edit either,
    or pl.endpoints' fallback layout would draw a graph qrot never solved over."""
    a = toy.copy()
    gr = dt.tl.endpoints(a, a, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                         p_end=0.05, verbose=0)
    before = _edges_with_data(gr["G"].to_networkx())
    a.obsp["neighbors_intr_distances"].data *= 2.0
    a.obsp["neighbors_intr_connectivities"].data *= 3.0
    # the edit did land: a graph built from the object now differs
    assert _edges_with_data(_build_digraph(a, graph="intr")) != before
    assert _edges_with_data(gr["G"].to_networkx()) == before


def test_graph_survives_deepcopy_and_pickle_on_a_view(toy):
    """Graph results get deep-copied and pickled (checkpoints, joblib). Built from an
    AnnData view, the stored matrices must be plain copies: anndata deep-copies a view's
    matrix into its parent's whole obsp, which to_networkx() cannot read."""
    view = toy[np.arange(toy.n_obs) % 7 != 0]
    assert view.is_view
    gr = dt.tl.endpoints(view, view, ["source"], ["sink"], cluster_key="clusters",
                         graph="intr", p_end=0.05, verbose=0)
    expect = _edges_with_data(_build_digraph(view, graph="intr"))
    assert _edges_with_data(gr["G"].to_networkx()) == expect
    for clone in (copy.deepcopy(gr["G"]), pickle.loads(pickle.dumps(gr["G"]))):
        assert clone.edge_array.tobytes() == gr["G"].edge_array.tobytes()
        assert clone.cost_vec.tobytes() == gr["G"].cost_vec.tobytes()
        assert _edges_with_data(clone.to_networkx()) == expect


def test_networkx_style_access_raises_and_names_the_fix(toy):
    """endpoints()['G'] is no longer a networkx.DiGraph. A networkx idiom must raise and
    point at to_networkx(), not run on an array: `(u, v) in G.edges` on the (m, 2) edge
    array was true for nearly any pair."""
    G = dt.tl.endpoints(toy, toy, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                        p_end=0.05, verbose=0)["G"]
    assert isinstance(G, dt.tl.CellGraph) and "CellGraph" in dt.tl.__all__
    for name, hint in (("edges", True), ("nodes", True), ("to_undirected", True),
                       ("number_of_edges", True), ("no_such_attribute", False)):
        with pytest.raises(AttributeError, match=name) as exc:
            getattr(G, name)
        assert (".to_networkx()" in str(exc.value)) is hint, str(exc.value)


@pytest.mark.parametrize("shape", [(1, -1), (-1, 1)])
def test_a_2d_demand_fails_the_same_way_on_either_graph(toy, shape):
    """qrot copies f into place element by element for both kinds of graph, so a demand
    of the wrong shape raises exactly what it raised on the networkx graph."""
    gr = dt.tl.endpoints(toy, toy, ["source"], ["sink"], cluster_key="clusters", graph="intr",
                         p_end=0.05, verbose=0)
    bad = gr["f"].reshape(shape)
    msgs = []
    for G in (gr["G"], gr["G"].to_networkx()):
        with pytest.raises(ValueError) as exc:
            dt.tl.qrot(dict(gr, G=G, f=bad), verbose=0)
        msgs.append(str(exc.value))
    assert msgs[0] == msgs[1]
