"""Which random stream each cluster samples from, and what that buys.

Every draw in ``endpoints`` comes from a stream keyed by ``(seed, role, cluster)`` rather than
from one stream shared by the whole call. Three properties follow.

Reordering a role's cluster list cannot move a single cell. Under the shared stream a cluster
inherited whatever state the clusters listed before it had left, so its sample was a function
of its POSITION: on a real section, reversing a 3-element sink list moved a large share of
the sink cells in both ``"random"`` and ``"directed"`` mode, at identical quotas.

Appending a cluster to a role cannot move the cells the others get. The same defect from the
other side, and the sharper probe of the two, since it holds the list order fixed.

``seed`` still moves everything it used to, which is the property the fix must not pay with.
That includes ``seed=None``, which the key would otherwise hash as the literal "None" and
freeze.

Both invariances are asserted per role, not across roles: under ``"directed"`` the sink anchor
is by design the sink cell farthest from any SOURCE center, so changing the source list is
meant to move the sinks. What must not happen is a source list change moving the sinks through
the RNG.

Which of the five draw sites is live depends on the configuration, and a config where none is
would pass every test here for free. So each config below declares the roles whose sample the
seed must move, and that is asserted. Two are worth spelling out. Sink ``"centered"`` and
``"directed"`` shells under the default ``directed_metric="cost"`` are stratified on float
geodesics, so the shells are singletons, the within-shell shuffle has nothing to permute and
the sink sample is deterministic; ``"hops"`` makes those shells wide and the shuffle live.
And ``_medoid`` only draws on a pool over 400 cells, which the line fixture exists to give it.

The quotas are pinned in ``test_p_basis`` and ``test_alloc_power``; here they are only
asserted to hold still, so a moved cell cannot be excused as a moved quota.
"""

import itertools
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from _line_toy import line_toy

import deltatrace as dt
from deltatrace._endpoints import _cluster_rng

SEEDS = (0, 42, 2024)

# name: (fixture, p_end, endpoints kwargs, roles whose sample the seed must move)
CONFIGS = {
    "random": ("toy", 0.2, {"sampling_mode": "random"}, {"source", "sink"}),
    "centered": ("toy", 0.2, {"sampling_mode": "centered"}, {"source"}),
    "directed": ("toy", 0.2, {"sampling_mode": "directed"}, {"source"}),
    "per-role": ("toy", 0.2, {"sampling_mode": {"source": "random", "sink": "directed"}},
                 {"source"}),
    "hop-shells": ("toy", 0.2, {"sampling_mode": "directed", "directed_metric": "hops"},
                   {"source", "sink"}),
    "medoid-subsample": ("line", 0.05, {"sampling_mode": "directed"}, {"sink"}),
}
ROLES = {
    "toy": (["source", "mid"], ["A", "B"], "C"),
    "line": (["src", "helper"], ["big", "small"], "spare"),
}
LINE_SIZES = {"src": 40, "helper": 40, "big": 500, "small": 60, "spare": 40}


def _toy_object(n=240):
    """The toy system, its terminal third split three ways so both roles are multi-cluster.

    Real geometry rather than a hand-built graph: the source hop shells come out ragged, which
    is what keeps the within-shell shuffle doing something.
    """
    a, d = line_toy(n=n, marker_noise=0.05)
    t = np.linspace(0, 1, n)
    i = np.arange(n)
    a.obs["clusters"] = pd.Categorical(
        np.where(t < 1/3, "source",
                 np.where(t < 2/3, "mid",
                          np.where(i % 3 == 0, "A", np.where(i % 3 == 1, "B", "C"))))
    )
    return dt.pp.integrate(a, d, alpha=0.9, p_keep=0.3, verbose=0)


def _line_object(sizes, k=6):
    """Object whose clusters have exactly the given eligible-pool sizes.

    Cells sit on one line and are joined by a ``k``-nearest-neighbor graph, so the whole object
    is one connected component and every cluster's pool equals its size. ``big`` is over 400
    cells, the threshold above which ``_medoid`` subsamples and therefore draws.
    """
    labels = [c for c, m in sizes.items() for _ in range(m)]
    n = len(labels)
    xy = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    D = np.abs(xy[:, None, 0] - xy[None, :, 0])
    np.fill_diagonal(D, np.inf)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in np.argsort(D[i])[:k]:
            rows.append(i)
            cols.append(int(j))
            vals.append(float(D[i, int(j)]))
    obj = ad.AnnData(
        X=np.zeros((n, 2), dtype=np.float32),
        obs=pd.DataFrame({"clusters": labels}, index=[f"c{i}" for i in range(n)]),
    )
    obj.obsm["spatial"] = xy
    obj.obsp["neighbors_intr_distances"] = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    obj.obsp["neighbors_intr_connectivities"] = sp.csr_matrix(
        (1.0 / (1.0 + np.asarray(vals)), (rows, cols)), shape=(n, n)
    )
    obj.uns["neighbors_intr"] = {
        "connectivities_key": "neighbors_intr_connectivities",
        "distances_key": "neighbors_intr_distances",
        "params": {"n_neighbors": k, "method": "umap"},
    }
    return obj


@pytest.fixture(scope="module")
def objects():
    return {"toy": _toy_object(), "line": _line_object(LINE_SIZES)}


def _sample(obj, src, snk, p_end, kwargs, seed=42):
    """Per-cluster cell lists for both roles, plus the demand vector.

    ``p_basis="cluster"`` unless a config overrides it: it sizes each cluster from its own
    pool, so the quotas are a function of the arguments and the data alone. Under ``"lcc"`` a
    role's total is split with a random remainder, which moves the quotas as the list is
    reordered and would hide what these tests are about.
    """
    gr = dt.tl.endpoints(obj, list(src), list(snk), cluster_key="clusters", graph="intr",
                         p_end=p_end, seed=seed, verbose=0,
                         **{"p_basis": "cluster", **kwargs})
    lab = obj.obs["clusters"].astype(str).to_numpy()
    out = []
    for key in ("source_indices", "sink_indices"):
        idx = np.asarray(gr[key], dtype=np.int64)
        out.append({c: [int(i) for i in idx[lab[idx] == c]] for c in np.unique(lab[idx])})
    return out[0], out[1], np.asarray(gr["f"], dtype=np.float64)


def _setup(objects, config):
    fixture, p_end, kwargs, live = CONFIGS[config]
    src, snk, extra = ROLES[fixture]
    return objects[fixture], src, snk, extra, p_end, kwargs, live


@pytest.mark.parametrize("config", CONFIGS)
def test_reordering_a_role_moves_nothing(objects, config):
    """Reverse the sources, the sinks, then both: same cells, same order, same demand vector."""
    obj, src, snk, _, p_end, kwargs, _ = _setup(objects, config)
    src0, snk0, f0 = _sample(obj, src, snk, p_end, kwargs)
    for s, k in ((src, snk[::-1]), (src[::-1], snk), (src[::-1], snk[::-1])):
        got_s, got_k, f = _sample(obj, s, k, p_end, kwargs)
        assert got_s == src0, f"source cells depend on list order: {s} -> {snk}"
        assert got_k == snk0, f"sink cells depend on list order: {s} -> {k}"
        assert np.array_equal(f, f0), "the demand vector depends on list order"


@pytest.mark.parametrize("mode", ["random", "directed"])
def test_reordering_a_role_moves_nothing_on_the_lcc_basis(objects, mode):
    """The same property on ``p_basis="lcc"`` (the default before 0.1.0, still supported).

    ``"lcc"`` hands a role's total out equally with the remainder drawn over the role's list,
    so reordering it can move a quota and with it a cell. At ``p_end=0.05`` on this
    fixture the total is 12, which divides evenly by both role sizes, the remainder loop makes
    no draw and the property holds outright: quotas first, so a moved cell cannot be excused as
    a moved remainder. Every permutation of both lists, not just the reversals, since three
    sinks make that cheap. This is why the basis cannot join ``CONFIGS``: appending a cluster
    changes every quota under ``"lcc"``, so the sibling test has nothing to compare.
    """
    obj = objects["toy"]
    src, snk = ["source", "mid"], ["A", "B", "C"]
    kwargs = {"sampling_mode": mode, "p_basis": "lcc"}
    src0, snk0, f0 = _sample(obj, src, snk, 0.05, kwargs)
    assert {c: len(v) for c, v in src0.items()} == {"source": 6, "mid": 6}
    assert {c: len(v) for c, v in snk0.items()} == {"A": 4, "B": 4, "C": 4}
    for s in itertools.permutations(src):
        for k in itertools.permutations(snk):
            got_s, got_k, f = _sample(obj, s, k, 0.05, kwargs)
            assert {c: len(v) for c, v in got_k.items()} == {c: len(v) for c, v in snk0.items()}, (
                f"sink quotas depend on list order: {s} -> {k}")
            assert got_s == src0, f"source cells depend on list order: {s} -> {k}"
            assert got_k == snk0, f"sink cells depend on list order: {s} -> {k}"
            assert np.array_equal(f, f0), "the demand vector depends on list order"


@pytest.mark.parametrize("config", CONFIGS)
def test_adding_a_cluster_to_a_role_leaves_the_others_alone(objects, config):
    """The property positional streams lack, tested at both ends of the list.

    At the FRONT is the case a shared stream cannot survive: the new cluster draws first and
    every cluster behind it inherits a different state. At the BACK it can survive, which is
    why the front matters. Per role: growing the sink list must not move a sink already in it,
    growing the source list must not move a source. Growing the SOURCE list is allowed to move
    the sinks, since the ``"directed"`` anchor is defined against the source centers.
    """
    obj, src, snk, extra, p_end, kwargs, _ = _setup(objects, config)
    src0, snk0, _ = _sample(obj, src, snk, p_end, kwargs)
    for grown_snk in ([extra, *snk], [*snk, extra]):
        got_s, got_k, _ = _sample(obj, src, grown_snk, p_end, kwargs)
        assert {c: got_k[c] for c in snk} == snk0, f"sinks moved for {grown_snk}"
        assert got_s == src0, f"sources moved for sinks {grown_snk}"
    for grown_src in ([extra, *src], [*src, extra]):
        got_s, _, _ = _sample(obj, grown_src, snk, p_end, kwargs)
        assert {c: got_s[c] for c in src} == src0, f"sources moved for {grown_src}"


@pytest.mark.parametrize("config", CONFIGS)
def test_seed_still_moves_the_sample(objects, config):
    """The seed keeps its reach, and each config's live draw sites are named rather than assumed.

    Without this the two invariance tests could pass on a configuration that draws nothing.
    """
    obj, src, snk, _, p_end, kwargs, live = _setup(objects, config)
    seen = {"source": set(), "sink": set()}
    for seed in SEEDS:
        got_s, got_k, _ = _sample(obj, src, snk, p_end, kwargs, seed=seed)
        seen["source"].add(tuple(sorted(i for v in got_s.values() for i in v)))
        seen["sink"].add(tuple(sorted(i for v in got_k.values() for i in v)))
    for role in live:
        assert len(seen[role]) == len(SEEDS), f"seed does not reach the {role} sample in {config}"


def test_the_stream_key_is_seed_role_and_cluster():
    """Same key, same stream; a different seed, role or label, a different one.

    The role is in the key because a cluster can be named as both a source and a sink, and the
    two samples must be independent draws rather than one draw used twice.
    """
    def draws(seed, role, cluster):
        rng = _cluster_rng(seed, role, cluster)
        return [rng.random() for _ in range(4)]

    assert draws(42, "source", "A") == draws(42, "source", "A")
    assert draws(42, "source", "A") != draws(42, "sink", "A")
    assert draws(42, "source", "A") != draws(42, "source", "B")
    assert draws(42, "source", "A") != draws(43, "source", "A")


def test_a_cluster_in_both_roles_gets_two_different_samples(objects):
    """End to end: 'mid' as source and as sink is not handed the same cells twice.

    ``"random"`` on both roles, where the two draws are otherwise identical in every respect
    except the role in the key, and ``p_end`` low enough that the two samples do not
    overlap: ``endpoints`` rejects an overlap outright, which would mask this.
    """
    src, snk, _ = _sample(objects["toy"], ["source", "mid"], ["mid", "A"], 0.05,
                          {"sampling_mode": "random"})
    assert len(src["mid"]) == len(snk["mid"]) == 4, "quota moved: this is no longer the claim"
    assert set(src["mid"]) != set(snk["mid"]), "'mid' draws from one stream in both roles"


def test_overlapping_both_role_samples_name_the_cluster(objects):
    """Two independent samples for one cluster can collide, and that must not be a bare assert.

    An overlap makes the demand vector unbalanceable, so it is the caller's problem to fix and
    the message has to say which cluster and how many cells. A bare ``assert`` reported it as a
    residual, and vanished under ``python -O`` leaving the solve to run on unbalanced demand.
    """
    with pytest.raises(ValueError, match=r"both a source and a sink.*'mid': \d+"):
        _sample(objects["toy"], ["source", "mid"], ["mid", "A"], 0.2,
                {"sampling_mode": "random"})


def test_seed_none_still_means_a_fresh_sample(objects):
    """``seed=None`` is out of contract but has always meant 'do not reproduce me'.

    The per-cluster key is built from the seed's string form, so an unresolved ``None`` would
    hash the literal and freeze every draw: the same cells every run, with no error, which
    reads as perfect stability to any bootstrap loop. ``endpoints`` resolves it to one drawn
    integer instead. Four runs colliding by chance is a 1-in-1e16 event on these quotas.
    """
    obj = objects["toy"]
    seen = set()
    for _ in range(4):
        got_s, _, _ = _sample(obj, ["source", "mid"], ["A", "B"], 0.2,
                              {"sampling_mode": "random"}, seed=None)
        seen.add(tuple(sorted(i for v in got_s.values() for i in v)))
    assert len(seen) == 4, f"seed=None repeats itself: {len(seen)} distinct samples in 4 runs"


def test_streams_are_reproducible_across_processes():
    """Two interpreters with different PYTHONHASHSEED must derive the same streams.

    The sub-seed is a hashlib digest of ``(seed, role, cluster)``, not Python's builtin
    ``hash()``, which is salted per process: with the builtin, which cells a cluster got would
    depend on the interpreter that sampled them and no run would be reproducible.
    """
    code = ("import deltatrace._endpoints as C;"
            "print([round(C._cluster_rng(42, r, c).random(), 12)"
            " for r in ('source', 'sink') for c in ('A', 'B')])")
    out = [
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                       env={"PYTHONHASHSEED": h, "PATH": "/usr/bin:/bin"}).stdout.strip()
        for h in ("0", "1", "random")
    ]
    assert out[0] == out[1] == out[2], f"streams depend on PYTHONHASHSEED: {out}"
    first = [float(x) for x in out[0].strip("[]").split(",")]
    assert len(set(first)) == 4, f"two (role, cluster) pairs share a stream: {out[0]}"
