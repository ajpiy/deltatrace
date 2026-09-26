"""Which geodesic the centered/directed sampler measures with, and who seeds its subsample.

Two things are pinned here.

``directed_metric`` picks the geodesic used by the sink medoid, the sink anchor and the
sink sampling shells. Under the default ``"cost"`` the sampled sinks move whenever the edge
costs move, so an ``alpha`` sweep re-samples every arm and never compares like with like.
Under ``"hops"`` they depend only on the edge set, which ``alpha`` does not touch on an
``intr`` graph. Source sampling never consults the option, which is pinned below too.

The medoid subsample is drawn from the cluster's own stream off ``endpoints``' ``seed``, not
from the process-global NumPy RNG. Clusters over 400 cells take that branch, so the tests
below build one.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from _line_toy import line_toy

import deltatrace as dt

ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)

# Sampled indices from before directed_metric existed, captured by running this exact
# configuration against a snapshot of the pre-change source. "cost" must reproduce them.
#
# The SINKS still hold literally, and are worth keeping for that reason. Their quota fills
# whole shells around the anchor, so which cells land in the sample is decided by the shell
# structure and not by the within-shell shuffle, and re-seeding that shuffle per (role,
# cluster) could not move them. The SOURCES did move, by 2 cells of 18 (5 in, 19 out): the
# source shells do end on a partly-consumed one, so the shuffle picks the last few, and that
# draw now comes from the 'source' cluster's own stream instead of the shared one. Both are
# pinned all the same, since drift in either is a regression.
PRECHANGE_SINKS = {
    0.1: [263, 272, 276, 279, 280, 283, 285, 287, 288, 289, 291, 292, 293, 294, 295, 297,
          298, 299],
    0.3: [252, 257, 258, 265, 267, 268, 269, 270, 271, 272, 275, 277, 278, 281, 282, 283,
          284, 287],
    0.5: [272, 279, 280, 283, 285, 286, 287, 288, 289, 291, 292, 293, 294, 295, 296, 297,
          298, 299],
    0.7: [279, 280, 281, 283, 285, 286, 287, 288, 289, 291, 292, 293, 294, 295, 296, 297,
          298, 299],
    0.9: [281, 282, 283, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297,
          298, 299],
}
SOURCES = [0, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 18, 20, 21, 22, 26, 28, 29]


@pytest.fixture(scope="module")
def alpha_graphs():
    """One toy graph per alpha. The edge set is identical across them; only costs move.
    p_keep=0.1 (the pre-0.1.0 default) is spelled out: PRECHANGE_SINKS were recorded on it."""
    return {a: dt.pp.graphs(*line_toy(n=300), alpha=a, p_keep=0.1, verbose=0)
            for a in ALPHAS}


def _sample(obj, metric):
    gr = dt.tl.endpoints(obj, obj, ["source"], ["sink"], cluster_key="clusters",
                         graph="intr", sampling_mode="directed", p_end=0.06,
                         seed=42, directed_metric=metric, verbose=0, p_basis="lcc")
    return sorted(map(int, gr["sink_indices"])), sorted(map(int, gr["source_indices"]))


def _two_lobe_object(n_snk=500, n_src=40, k=8, seed_geom=0):
    """A source cluster plus a two-lobed sink cluster of more than 400 cells.

    Two lobes, because the medoid of a single round blob survives almost any subsample;
    with two the winner depends on which cells the 400-cell draw happens to include.

    Parameters
    ----------
    n_snk, n_src : int
        Cells in the sink and source clusters. ``n_snk`` must exceed 400 to reach the
        subsampling branch of the medoid.
    k : int, default 8
        Neighbors per cell in the hand-built kNN graph.
    seed_geom : int, default 0
        Seed for the point layout only. Unrelated to the sampling seed under test.

    Returns
    -------
    anndata.AnnData
        Object carrying ``obs['clusters']`` and a ``neighbors_intr`` graph.
    """
    rng = np.random.default_rng(seed_geom)
    xy_src = rng.normal(0.0, 1.0, size=(n_src, 2))
    half = n_snk // 2
    lobe_a = rng.normal(0.0, 1.5, size=(half, 2)) + np.array([4.0, 4.0])
    lobe_b = rng.normal(0.0, 1.5, size=(n_snk - half, 2)) + np.array([8.0, -4.0])
    xy = np.vstack([xy_src, lobe_a, lobe_b])
    n = n_src + n_snk

    D = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in np.argsort(D[i])[:k]:
            rows.append(i)
            cols.append(int(j))
            vals.append(float(D[i, int(j)]))
    vals_arr = np.asarray(vals)
    dist = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    conn = sp.csr_matrix((1.0 / (1.0 + vals_arr), (rows, cols)), shape=(n, n))

    obj = ad.AnnData(
        X=np.zeros((n, 2), dtype=np.float32),
        obs=pd.DataFrame({"clusters": ["source"] * n_src + ["sink"] * n_snk},
                         index=[f"c{i}" for i in range(n)]),
    )
    obj.obsm["spatial"] = xy
    obj.obsp["neighbors_intr_distances"] = dist
    obj.obsp["neighbors_intr_connectivities"] = conn
    obj.uns["neighbors_intr"] = {
        "connectivities_key": "neighbors_intr_connectivities",
        "distances_key": "neighbors_intr_distances",
        "params": {"n_neighbors": k, "method": "umap"},
    }
    return obj


@pytest.fixture(scope="module")
def big_cluster():
    return _two_lobe_object()


# Every legacy global-RNG sampler a subsample could plausibly be rewritten to use. The
# defect was never `choice` in particular, it was drawing from this shared stream at all,
# so the guard covers the family rather than the one call that happened to be there.
_GLOBAL_SAMPLERS = ("choice", "shuffle", "permutation", "randint", "random_integers",
                    "random", "random_sample", "rand", "ranf", "sample", "bytes",
                    "uniform", "normal", "integers")


@pytest.fixture
def no_global_numpy_draw(monkeypatch):
    """Make any draw from the process-global NumPy RNG an error.

    The medoid subsample must come from ``endpoints``' ``seed``, so nothing on this path
    may reach for ``np.random``. ``np.random.seed`` is deliberately left alone: seeding
    that stream is harmless, consuming it is the defect.
    """
    def boom(name):
        def _boom(*args, **kwargs):
            raise RuntimeError(
                "medoid subsample must not draw from the global NumPy RNG "
                f"(np.random.{name})"
            )
        return _boom

    patched = 0
    for name in _GLOBAL_SAMPLERS:
        if hasattr(np.random, name):
            monkeypatch.setattr(np.random, name, boom(name))
            patched += 1
    assert patched >= 8, f"guard covers too little of np.random: {patched}"


def _sink_center(obj, seed):
    """The medoid the sampler centered the sink shells on, read off the level-2 log."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gr = dt.tl.endpoints(obj, obj, ["source"], ["sink"], cluster_key="clusters",
                             graph="intr", p_end=0.02, seed=seed, verbose=2,
                             sampling_mode={"source": "random", "sink": "centered"})
    lines = [ln for ln in buf.getvalue().splitlines() if "Sink center for" in ln]
    assert len(lines) == 1, buf.getvalue()
    return lines[0].split(":")[-1].strip(), sorted(map(int, gr["sink_indices"]))


def test_cost_metric_reproduces_prechange_sinks(alpha_graphs):
    """The default metric samples the sinks the code sampled before the option existed.

    The source list is pinned too, but against the current tree: per-(role, cluster) RNG
    streams moved 2 of its 18 cells, where the sinks are shuffle-independent and did not.
    """
    for alpha in ALPHAS:
        sinks, sources = _sample(alpha_graphs[alpha], "cost")
        assert sinks == PRECHANGE_SINKS[alpha], f"sink drift at alpha={alpha}"
        assert sources == SOURCES, f"source drift at alpha={alpha}"


def test_hops_metric_holds_endpoints_fixed_across_alpha(alpha_graphs):
    """'hops' pins the endpoint set across an alpha sweep where 'cost' moves it."""
    # The edge set is what "hops" measures on, so it must not be moving either.
    edges = {int(alpha_graphs[a].obsp["neighbors_intr_distances"].nnz) for a in ALPHAS}
    assert len(edges) == 1, f"alpha changed the edge set: {edges}"

    hop_sets = [_sample(alpha_graphs[a], "hops")[0] for a in ALPHAS]
    assert all(s == hop_sets[0] for s in hop_sets), "hops sampling moved with alpha"

    cost_sets = [_sample(alpha_graphs[a], "cost")[0] for a in ALPHAS]
    assert any(s != cost_sets[0] for s in cost_sets), "premise gone: cost is alpha-stable"
    # And the two metrics really are different instruments, not the same one renamed.
    assert hop_sets[0] != cost_sets[0]


def test_directed_metric_is_sink_only(alpha_graphs):
    """The option reaches the sink sample and nothing else, as its docstring now says.

    Source ``"centered"`` routes through a sampler that is unweighted whatever the option
    says, so a mode that centers only the source must be completely inert.
    """
    obj = alpha_graphs[0.5]

    def both(mode):
        out = {}
        for m in ("cost", "hops"):
            gr = dt.tl.endpoints(obj, obj, ["source"], ["sink"], cluster_key="clusters",
                                 graph="intr", sampling_mode=mode, p_end=0.06,
                                 seed=42, directed_metric=m, verbose=0)
            out[m] = (sorted(map(int, gr["source_indices"])),
                      sorted(map(int, gr["sink_indices"])))
        return out

    # A centered SOURCE with a random sink: a role does sample "centered", yet nothing moves.
    inert = both({"source": "centered", "sink": "random"})
    assert inert["cost"] == inert["hops"], "directed_metric leaked into source sampling"

    # A centered SINK does move, so the inertness above is scope, not a no-op option.
    live = both({"source": "centered", "sink": "centered"})
    assert live["cost"][1] != live["hops"][1], "premise gone: sink sampling is metric-blind"
    assert live["cost"][0] == live["hops"][0], "directed_metric leaked into source sampling"


def test_blocked_flows_does_not_move_the_endpoints(alpha_graphs):
    """``blocked_flows`` reprices edges after sampling, so it cannot move the endpoints.

    The parameter docstring says so explicitly; this pins it under both metrics.
    """
    obj = alpha_graphs[0.5]
    blocked = [("sink", "mid"), ("mid", "source"), ("sink", "source")]

    def go(metric, bf):
        gr = dt.tl.endpoints(obj, obj, ["source"], ["sink"], cluster_key="clusters",
                             graph="intr", sampling_mode="directed", p_end=0.06,
                             seed=42, directed_metric=metric, blocked_flows=bf, verbose=0)
        return ((sorted(map(int, gr["source_indices"])),
                 sorted(map(int, gr["sink_indices"]))), float(gr["cost_vec"].sum()))

    for metric in ("cost", "hops"):
        plain_ep, plain_cost = go(metric, None)
        blocked_ep, blocked_cost = go(metric, blocked)
        assert plain_ep == blocked_ep, f"blocked_flows moved the endpoints ({metric})"
        # And it did reach the costs, so the equality above is not a dead comparison.
        assert plain_cost != blocked_cost, f"blocked_flows never applied ({metric})"


@pytest.mark.parametrize("bad", ["hop", "Cost", "dijkstra", ""])
def test_unknown_directed_metric_raises_value_error(alpha_graphs, bad):
    """A misspelled metric is a caller error, named as such, whatever the sampling mode."""
    obj = alpha_graphs[0.5]
    with pytest.raises(ValueError, match=r"directed_metric") as exc:
        dt.tl.endpoints(obj, obj, ["source"], ["sink"], cluster_key="clusters",
                        graph="intr", sampling_mode="directed", p_end=0.06,
                        directed_metric=bad, verbose=0)
    msg = str(exc.value)
    assert "'cost'" in msg and "'hops'" in msg


def test_medoid_subsample_is_reproducible_from_seed(big_cluster, no_global_numpy_draw):
    """More than 400 eligible cells: one seed, one medoid, call after call."""
    pool = int((big_cluster.obs["clusters"] == "sink").sum())
    assert pool > 400, "fixture no longer reaches the subsampling branch"

    first = _sink_center(big_cluster, seed=0)
    for _ in range(3):
        assert _sink_center(big_cluster, seed=0) == first


def test_medoid_subsample_moves_with_seed(big_cluster, no_global_numpy_draw):
    """The same seed argument that fixes the medoid can also change it."""
    centers = {_sink_center(big_cluster, seed=s)[0] for s in range(8)}
    assert len(centers) > 1, f"seed had no effect on the medoid: {centers}"


def test_medoid_subsample_leaves_the_global_numpy_stream_untouched(big_cluster):
    """Sampler-agnostic form of the same property: the shared stream is not consumed.

    ``endpoints`` reseeds the global NumPy RNG on entry, so if it draws nothing the state
    afterwards is exactly the state a bare reseed leaves. Unlike the monkeypatch guard this
    holds whichever sampler a regression might reach for.
    """
    seed = 0
    _sink_center(big_cluster, seed=seed)
    after_call = np.random.get_state()

    np.random.seed(seed)
    after_bare_reseed = np.random.get_state()

    assert after_call[0] == after_bare_reseed[0]
    assert np.array_equal(after_call[1], after_bare_reseed[1])
    assert after_call[2:] == after_bare_reseed[2:], "something drew from np.random"


def test_medoid_survives_a_third_party_draw_mid_call(big_cluster, monkeypatch):
    """A dependency drawing from the global NumPy RNG during the call must not be felt.

    This is what the old ``np.random.choice`` subsample was actually exposed to: reseeding
    a shared stream on entry only holds the draw fixed while nothing else consumes it.
    A stand-in for such a dependency is spliced into graph construction here: the
    component search endpoints runs before any sampling.
    """
    import scipy.sparse.csgraph as csg

    real = csg.connected_components
    strays = (0, 1, 3, 7, 11, 19, 31)
    centers = {}
    reached = []
    for stray in strays:
        def noisy(*a, _k=stray, **kw):
            reached.append(_k)
            np.random.random(_k)
            return real(*a, **kw)

        monkeypatch.setattr(csg, "connected_components", noisy)
        centers[stray] = _sink_center(big_cluster, seed=0)[0]

    # Without this the test passes vacuously: every call must have gone through the draw.
    assert reached == list(strays), f"the injected draw was not on the call path: {reached}"
    assert len(set(centers.values())) == 1, f"a third-party draw moved the medoid: {centers}"


def test_trajectory_forwards_directed_metric():
    """trajectory hands directed_metric to endpoints rather than dropping it."""
    obj = dt.pp.graphs(*line_toy(n=300), alpha=0.5, verbose=0)
    got = {}
    for metric in ("cost", "hops"):
        a = obj.copy()
        res = dt.tl.trajectory(a, ["source"], ["sink"], cluster_key="clusters",
                               graph="intr", basis="spatial", p_end=0.06,
                               sampling_mode="directed", directed_metric=metric,
                               compute_velocity=False, verbose=0, p_basis="lcc")
        got[metric] = sorted(map(int, res["endpoints"]["sink_indices"]))
        assert got[metric] == _sample(obj, metric)[0]
    assert got["cost"] != got["hops"]        # the argument reached far enough to matter
