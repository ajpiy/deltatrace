"""Edge costs must be finite and non-negative, and a bad one must fail loudly at graph
build: scipy's Dijkstra (endpoint sampling) returns arbitrary results on NaN and does not
terminate on a negative edge, and the transport solve would run on garbage."""

import numpy as np
import pytest
from _line_toy import line_toy

import deltatrace as dt
from deltatrace._utils import _build_cell_graph, _build_digraph


@pytest.fixture
def graph():
    adata, adata_dist = line_toy(n=90)
    return dt.pp.graphs(adata, adata_dist, alpha=0.9, p_keep=0.3, verbose=0)


def _poison(g, value, k=3):
    key = g.uns["neighbors_intr"]["distances_key"]
    D = g.obsp[key].tocsr(copy=True)
    D.data[:k] = value
    g.obsp[key] = D
    return key


@pytest.mark.parametrize("bad, word", [(np.nan, "3 non-finite"), (np.inf, "3 non-finite"),
                                       (-0.5, "3 negative")])
def test_bad_costs_are_refused_by_both_builders(graph, bad, word):
    key = _poison(graph, bad)
    for build in (_build_cell_graph, _build_digraph):
        with pytest.raises(ValueError, match=word) as exc:
            build(graph, graph="intr")
        assert key in str(exc.value)


def test_bad_costs_are_refused_before_endpoint_sampling(graph):
    """The user-facing entry: tl.endpoints must raise before any Dijkstra runs."""
    _poison(graph, -1.0, k=1)
    with pytest.raises(ValueError, match="1 negative"):
        dt.tl.endpoints(graph, graph, ["source"], ["sink"], cluster_key="clusters",
                        graph="intr", p_end=0.2, verbose=0)


def test_valid_costs_pass_and_zero_is_allowed(graph):
    """A stored zero is not refused: eliminate_zeros drops it like an absent edge. Only
    NaN, inf and negative values are errors."""
    key = _poison(graph, 0.0, k=2)
    G = _build_cell_graph(graph, graph="intr")
    assert np.isfinite(G.cost_vec).all() and (G.cost_vec >= 0).all()
    assert key in graph.obsp
