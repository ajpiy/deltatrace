"""tl: lineages and terminality on the absorbing chain, widest_path, and the flow's cluster graph (cluster_trajectory).

The chain / cluster-graph machinery (build_chain, absorption, cluster_graph, ...) sits
below the public wrappers in this same module."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from . import _log
from ._utils import _suffix


# ---------------------------------------------------------------------------
# tl.lineages  (absorbing-chain fate assignment)
# ---------------------------------------------------------------------------
def _check_terminal_cut(terminal_clusters, omega_term, n_lineages) -> None:
    """Validate that omega_term/n_lineages are mutually exclusive, n_lineages is
    only used in source-only mode, and n_lineages (if given) is an integer >= 1.
    Shared by lineages() and trajectory() so a bad combination is rejected before
    a solve is spent on it."""
    if n_lineages is None:
        return
    if omega_term is not None:
        raise ValueError(
            f"Pass omega_term or n_lineages, not both (got {omega_term} and {n_lineages})."
        )
    if terminal_clusters is not None:
        raise ValueError(
            "n_lineages only applies in source-only mode, but terminal_clusters="
            f"{[str(c) for c in terminal_clusters]} already names the terminals."
        )
    if (isinstance(n_lineages, bool) or not isinstance(n_lineages, (int, np.integer))
            or n_lineages < 1):
        raise ValueError(f"n_lineages must be an integer >= 1, got {n_lineages!r}.")


@_log.stage("assigning lineages by absorbing chain")
def lineages(
    adata,
    source_clusters,
    terminal_clusters=None,
    *,
    cluster_key: str = "clusters",
    flow_key: str | None = None,
    pt_key: str | None = None,
    omega_term: float | None = None,
    n_lineages: int | None = None,
    edge_threshold: float = 0,
    verbose: int | bool = 1,
    key_added: str | None = None,
) -> dict:
    """Give every cell its probability of ending in each terminal fate.

    Builds the absorbing Markov chain from the QROT flow in ``adata.uns[flow_key]``
    and reads off the absorption probability that a unit of flow leaving cell ``i``
    ends in lineage ``l`` (unique and decomposition-independent, so a bipotent cell
    is reported as a split, e.g. 0.67 / 0.33, not committed to one fate).

    Writes the canonical keys tradeSeq reads (``obsm['path_cellweights']``,
    ``obsm['path_pseudotime']``, ``obsm['path_flowmass']``) plus ``obs['lineage']``
    / ``obs['lineage_confidence']`` / ``obs['fate_leakage']`` and one
    ``obs['fate_<lineage>']`` column per lineage (its absorption probability, for
    ``scanpy.pl.embedding(color=...)``; a ``/`` in the lineage name becomes ``_``). A cell the flow never
    reaches is ``"unassigned"`` in ``lineage`` and NaN in the numeric columns.

    Parameters
    ----------
    adata : anndata.AnnData
        Object QROT wrote into (holds ``uns[flow_key]`` and ``obs[pt_key]``).
    source_clusters : sequence of str
        Cluster labels (in ``obs[cluster_key]``) whose cells emit the flow.
    terminal_clusters : sequence of str, optional
        Terminal (sink) cluster labels. ``None`` runs SOURCE-ONLY discovery: ranks
        clusters by terminality (see :func:`terminality`) and cuts that ranking by
        ``omega_term`` or ``n_lineages`` (exactly one required).
    omega_term : float, optional
        Terminal cutoff on cluster terminality, ``term_score = 1 - flow_out/flow_in``; used only
        when ``terminal_clusters is None``. :func:`terminality` suggests a value
        from the largest gap in the sorted table.
    n_lineages : int, optional
        Take the top ``n_lineages`` clusters by terminality instead of a value
        cutoff. Source-only only, mutually exclusive with ``omega_term``.
    cluster_key : str, default "clusters"
        ``obs`` column holding the cluster labels.
    flow_key, pt_key : str
        ``uns`` / ``obs`` keys for the flow and pseudotime.
    edge_threshold : float, default 0
        Minimum flow for an edge to count; 0 keeps exactly the edges the optimal flow uses.
    verbose : int or bool, default 1
        Logging verbosity.
    key_added : str, optional
        Suffix for the keys read and written, matching the one given to :func:`qrot`.

    Returns
    -------
    dict
        ``"chain"`` (the absorbing chain), ``"A"`` (unconditional absorption per
        cell and lineage), ``"b"`` (probability of reaching any terminal), ``"F"``
        (conditional fate, ``A`` renormalized by ``b``), ``"names"`` (lineage
        names, column order of ``A``).
    """
    _check_terminal_cut(terminal_clusters, omega_term, n_lineages)
    _sfx = _suffix(key_added)
    flow_key = flow_key or f"dt_flow{_sfx}"
    pt_key = pt_key or f"pseudotime{_sfx}"

    clu = adata.obs[cluster_key].astype(str).to_numpy()
    src = [str(c) for c in source_clusters]
    source_idx = np.where(np.isin(clu, src))[0]
    if source_idx.size == 0:
        raise ValueError(
            f"No source cells: none of {src} found in obs['{cluster_key}']."
        )
    if terminal_clusters is None:
        # SOURCE-ONLY fallback: discover terminal clusters from the flow.
        term = cluster_terminality(
            adata, None, cluster_key, flow_key=flow_key, edge_threshold=edge_threshold,
            source_clusters=src,
        )
        if len(term) == 0:
            raise ValueError(
                "No cluster receives flow from another cluster, so there are no terminals "
                "to rank: check source_clusters and cluster_key."
            )
        if omega_term is None and n_lineages is None:
            omega, _ = suggest_omega(term["term_score"].to_numpy())
            picked = [str(c) for c in term.index[term["term_score"] >= omega]]
            raise ValueError(
                "Source-only lineages need n_lineages or omega_term; the terminality "
                f"ranking suggests {picked} (n_lineages={len(picked)}, "
                f"omega_term={omega:.3f})."
            )
        if n_lineages is not None:
            if n_lineages > len(term):
                raise ValueError(
                    f"n_lineages={n_lineages} exceeds the {len(term)} cluster(s) in the "
                    f"terminality ranking, {[str(c) for c in term.index]}."
                )
            terminal_clusters, tied = top_terminals(term, n_lineages)
            if tied:
                warnings.warn(
                    f"n_lineages={n_lineages} splits clusters {tied}, tied at term_score="
                    f"{float(term.loc[tied[0], 'term_score']):.6g}: pick an n_lineages "
                    "that keeps or drops them together.",
                    UserWarning,
                    stacklevel=3,
                )
            _log.detail(
                f"discovered terminals (top {n_lineages} by terminality): {terminal_clusters}"
            )
        else:
            terminal_clusters = [str(c) for c in term.index[term["term_score"] >= omega_term]]
            if not terminal_clusters:
                raise ValueError(
                    f"No cluster has term_score >= omega_term={omega_term} "
                    f"(max term_score = {term['term_score'].max():.3f}). Lower omega_term."
                )
            _log.detail(
                f"discovered terminals (term_score >= {omega_term}): {terminal_clusters}"
            )
    return lineages_from_chain(
        adata, source_idx, terminal_clusters, cluster_key=cluster_key,
        flow_key=flow_key, pt_key=pt_key, edge_threshold=edge_threshold,
        suffix=_sfx,
    )


def terminality(
    adata,
    *,
    cluster_key: str = "clusters",
    source_clusters=None,
    flow_key: str | None = None,
    edge_threshold: float = 0,
    key_added: str | None = None,
    show: bool = True,
):
    """Rank clusters by how terminal the flow makes them look.

    Scores each cluster by cross-cluster flux, terminality, ``term_score = 1 - flow_out / flow_in``: a
    cluster that takes flow in and passes none on scores near 1, a progenitor
    scores strongly negative. Prints the sorted table and suggests the clusters above
    its largest gap as terminals; pass their number to
    ``lineages(terminal_clusters=None, n_lineages=...)``, or cut the same ranking by
    value with ``omega_term=...``.

    Parameters
    ----------
    adata : anndata.AnnData
        Object holding the solved flow in ``uns[flow_key]``.
    cluster_key : str, default "clusters"
        ``obs`` column holding the cluster labels.
    source_clusters : sequence of str, optional
        Progenitor cluster label(s) to drop from the table before the gap search
        (sources score strongly negative and would otherwise hijack the
        suggestion). Pass the same value you give :func:`lineages`.
    flow_key : str, optional
        ``uns`` key holding the solved flow. Overrides ``key_added`` when given.
    edge_threshold : float, default 0
        Minimum flow for an edge to count; 0 keeps exactly the edges the optimal flow uses.
    key_added : str, optional
        Read ``uns['dt_flow_<key_added>']`` instead of the flat ``uns['dt_flow']``.
    show : bool, default True
        Print the table + suggestion (its own knob, not ``verbose``, since the
        table is this function's product). The DataFrame is returned regardless.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``term_score`` descending; columns ``flow_in``, ``flow_out``, ``absorbed``, ``term_score`` (terminality).
    """

    flow_key = flow_key or f"dt_flow{_suffix(key_added)}"
    g = cluster_terminality(
        adata, None, cluster_key, flow_key=flow_key, edge_threshold=edge_threshold,
        source_clusters=source_clusters,
    )
    if show:
        omega, info = suggest_omega(g["term_score"].to_numpy())
        print("cluster terminality (term_score = 1 - flow_out/flow_in, higher = more terminal):")
        print(g[["flow_in", "flow_out", "absorbed", "term_score"]].round(4).to_string())
        above = [str(c) for c in g.index[g["term_score"] >= omega]]
        if above:
            gap = "clear gap" if info["clear"] else "no clear gap, largest"
            print(f"\nsuggested terminals: {above} (n_lineages={len(above)}; {gap} "
                  f"{info['gap_lo']:.3f} -> {info['gap_hi']:.3f})")
    return g


def widest_path(adata, src, dst, *, flow_key: str | None = None,
                edge_threshold: float = 0, key_added: str | None = None):
    """Trace the highest-capacity route from one cell to another through the flow.

    Returns the ``src`` -> ``dst`` path whose thinnest edge is as thick as possible
    (max-bottleneck / "widest" path), over edges carrying more than
    ``edge_threshold``. Not a pipeline stage: :func:`lineages` assigns fates
    without building a path, so use this only for an explicit route between two
    cells.

    Parameters
    ----------
    adata : anndata.AnnData
        Object holding the solved flow in ``uns[flow_key]``.
    src, dst : int
        Row indices of the start and end cell in ``adata``.
    flow_key : str, optional
        ``uns`` key of the flow written by :func:`qrot`. Overrides ``key_added``
        when given.
    edge_threshold : float, default 0
        Minimum flow for an edge to count; 0 keeps exactly the edges the optimal flow uses.
    key_added : str, optional
        Namespace of the run to trace, matching the one given to :func:`qrot`.

    Returns
    -------
    path : list of int
        Cell indices along the route, empty when ``dst`` is unreachable.
    capacity : float
        Flow on the route's thinnest edge, 0.0 if there is no route.
    """

    flow_key = flow_key or f"dt_flow{_suffix(key_added)}"
    fl = adata.uns.get(flow_key)
    if fl is None:
        raise ValueError(
            f"No flow data in adata.uns['{flow_key}']. Run qrot() with adata_basis set "
            f"(and the same key_added) first."
        )
    E = np.asarray(fl["edges"], np.int64).reshape(-1, 2)   # an edgeless flow stores shape (0,)
    J = np.asarray(fl["J"], float).ravel()
    keep = J > edge_threshold
    # widest_path only needs the cell count; no chain or flow conservation required.
    return _widest_path({"n": adata.n_obs}, E[keep], J[keep], [src], [dst])


EPS = 1e-12
FATE_FLOOR = 1e-10   # fate shares below this are solver rounding (observed <= 7e-14; real >= 3e-4)


# ---------------------------------------------------------------------------
# Chain construction and absorption
# ---------------------------------------------------------------------------
def build_chain(adata, source_idx, flow_key="dt_flow", edge_threshold=0):
    """Turn the solved QROT flow into an absorbing Markov chain over cells.

    Keeps only edges carrying real flow (``J > edge_threshold``), then splits each
    cell's throughput between its outgoing edges and the absorbing state. Cells
    are identified by positional index, so rows of ``T`` line up with
    ``adata.obs_names``.

    Parameters
    ----------
    adata : anndata.AnnData
        Object holding the flow in ``uns[flow_key]``.
    source_idx : array-like of int
        Positions of the progenitor cells the demand vector emits from; no other
        cell may carry net supply.
    flow_key : str, default "dt_flow"
        ``uns`` key of the QROT flow, ``{J (m,), edges (m, 2)}``.
    edge_threshold : float, default 0
        Minimum flow for an edge to count; 0 keeps exactly the edges the optimal flow uses.

    Returns
    -------
    dict
        ``T`` (n, n) sparse transition matrix, ``q`` (n,) absorption probability,
        ``theta`` (n,) throughput, ``inflow``/``outflow``/``supply``/``absorbed``
        (n,), ``live`` (n,) cells the flow reaches, and ``n``.
    """
    fl = adata.uns[flow_key]
    J = np.asarray(fl["J"], float).ravel()
    E = np.asarray(fl["edges"], np.int64)
    n = adata.n_obs
    keep = J > edge_threshold  # active support (edges carrying real flow)
    u, v, j = E[keep, 0], E[keep, 1], J[keep]

    inflow  = np.bincount(v, j, minlength=n)
    outflow = np.bincount(u, j, minlength=n)
    src_i = np.asarray(source_idx, np.int64)
    net_emit = np.maximum(outflow - inflow, 0.0)  # supply needed to conserve

    # Fold the solve's non-source supply into sigma once negligible: a coding/solver
    # artifact should not silently corrupt every fate derived from this chain.
    resid = net_emit.copy(); resid[src_i] = 0.0  # non-source (spurious) supply
    src_supply = float(net_emit[src_i].sum())
    if resid.sum() > 1e-3 * max(src_supply, EPS):
        raise ValueError(
            f"The flow does not match these sources: non-source cells emit "
            f"{resid.sum():.2e} against {src_supply:.2e} from the sources. Pass the source "
            "clusters the flow was solved with.")
    sigma = net_emit  # sources + absorbed residual
    theta = inflow + sigma  # == outflow + delta
    delta = np.maximum(theta - outflow, 0.0)  # absorbed mass

    live = theta > EPS  # cells the flow reaches
    theta_safe = np.where(live, theta, 1.0)  # guard: 0/0 on unreached cells
    T = sp.csr_matrix((j / theta_safe[u], (u, v)), shape=(n, n))
    q = np.where(live, delta / theta_safe, 0.0)

    rs = np.asarray(T.sum(1)).ravel() + q  # rows(T) + q, exact by design
    bad = np.abs(rs[live] - 1.0).max() if live.any() else 0.0  # a breach is a coding error
    assert bad < 1e-8, f"rows(T)+q != 1 (max dev {bad:.2e}): chain construction error"
    return dict(T=T, q=q, theta=theta, inflow=inflow, outflow=outflow, supply=sigma,
                absorbed=delta, live=live, n=n)


def absorption(chain, groups):
    """Compute how each cell's descendants divide between the terminal groups.

    A cell reaches lineage l either by stopping where it is (if it is one of that lineage's
    terminal cells) or by stepping to a downstream cell, so ``A`` solves the linear system

        a[i,l] = q_i * 1{i in l} + sum_u T[i,u] a[u,l],   i.e.   (I - T) A = R

    exactly, cycles in the active flow included.

    Parameters
    ----------
    chain : dict
        Output of :func:`build_chain`.
    groups : dict of {str: array-like}
        Terminal cells per lineage, as a boolean mask or an index array over cells.

    Returns
    -------
    A : ndarray of shape (n, L)
        Unconditional absorption ``a[i,l]``: the share of the cell's mass that ends up in
        lineage l.
    b : ndarray of shape (n,)
        Row sums of ``A``, the probability of reaching ANY declared terminal. ``1 - b`` is
        the leakage: progeny that stop where no lineage was declared.
    F : ndarray of shape (n, L)
        Conditional fate ``a[i,l] / b_i``, i.e. given that the cell reaches a terminal at
        all, how its progeny split between lineages. Rows sum to 1; cells the flow never
        reaches are NaN.
    """
    n, T, q = chain["n"], chain["T"], chain["q"]
    names = list(groups)
    R = np.zeros((n, len(names)))
    for k, name in enumerate(names):  # absorbed mass q at this lineage's own terminal cells
        g = np.asarray(groups[name]); R[g, k] = q[g]
    A = spsolve(sp.identity(n, format="csc") - T, R).reshape(n, len(names))
    # shares below FATE_FLOOR of a cell's absorbed mass are rounding from the solve, not fates
    A[A < FATE_FLOOR * A.sum(1, keepdims=True)] = 0.0
    b = A.sum(1)
    F = np.divide(A, b[:, None], out=np.full_like(A, np.nan), where=b[:, None] > EPS)
    return A, b, F


# ---------------------------------------------------------------------------
# Terminal discovery  (the source-only mode)
# ---------------------------------------------------------------------------
def _rank_order(term):
    """Sort a terminality table: term_score descending, then absorbed descending, then cluster
    name ascending (a total order, so ties are deterministic)."""
    g = term.copy()
    g["_name"] = [str(i) for i in g.index]
    g = g.sort_values(["term_score", "absorbed", "_name"], ascending=[False, False, True])
    return g.drop(columns="_name")


def _cross_cluster_flux(adata, cluster_key, flow_key="dt_flow", edge_threshold=1e-6):
    """Active flow edges whose two ends carry different cluster labels, as a
    DataFrame with src/dst/j columns (directed, within-cluster edges dropped).
    The single place this edge frame is built: cluster_terminality sums it,
    cluster_flux_matrix pivots it, so the two can't drift apart."""
    import pandas as pd
    fl = adata.uns[flow_key]
    J = np.asarray(fl["J"], float).ravel(); E = np.asarray(fl["edges"], np.int64)
    k = J > edge_threshold
    codes, labs = pd.factorize(adata.obs[cluster_key].astype(str).to_numpy())
    cu, cv = codes[E[k, 0]], codes[E[k, 1]]
    cross = cu != cv
    return pd.DataFrame(dict(src=labs[cu[cross]], dst=labs[cv[cross]], j=J[k][cross]))


def cluster_terminality(adata, chain, cluster_key, flow_key="dt_flow", edge_threshold=0,
                        source_clusters=None):
    """Rank clusters by how much of the flow they swallow, i.e. how terminal they look.

        term_score_k = absorbed_k / flow_in_k = 1 - flow_out_k / flow_in_k        on CROSS-cluster flux only

    Print the sorted table and look for a gap before fixing omega_term.

    Parameters
    ----------
    adata : anndata.AnnData
        Object holding the flow in ``uns[flow_key]`` and labels in ``obs[cluster_key]``.
    chain : dict or None
        Unused; accepted so the chain helpers share one call signature.
    cluster_key : str
        ``obs`` column with the cluster labels.
    flow_key : str, default "dt_flow"
        ``uns`` key of the QROT flow.
    edge_threshold : float, default 0
        Minimum flow for an edge to count; 0 keeps exactly the edges the optimal flow uses.
    source_clusters : sequence of str, optional
        Progenitor labels to drop from the table before the gap search (they score
        strongly negative and would otherwise hijack :func:`suggest_omega`).

    Returns
    -------
    pandas.DataFrame
        One row per cluster, sorted by ``term_score`` descending; columns ``flow_in``
        (cross-cluster inflow), ``flow_out`` (outflow), ``absorbed`` and ``term_score`` (terminality). Clusters no
        cross-cluster flux reaches are absent.
    """
    import pandas as pd
    # SUM the shared cross-cluster edge frame; cluster_flux_matrix PIVOTS the same rows.
    df = _cross_cluster_flux(adata, cluster_key, flow_key=flow_key,
                             edge_threshold=edge_threshold)
    out_i = df.groupby("src")["j"].sum()
    in_i = df.groupby("dst")["j"].sum()
    g = pd.DataFrame(dict(flow_in=in_i, flow_out=out_i)).fillna(0.0)
    # term_score is only defined where mass arrives (drops the flow_in == 0 rows)
    g = g[g["flow_in"] > EPS].copy()
    if source_clusters is not None:
        src = {str(c) for c in source_clusters}
        g = g[~g.index.isin(src)].copy()
    g["absorbed"] = g["flow_in"] - g["flow_out"]
    g["term_score"] = 1.0 - g["flow_out"] / g["flow_in"]
    return _rank_order(g)


def suggest_omega(term_values):
    """Suggest where to cut the terminality ranking into terminal and non-terminal clusters.

    Sorts the scores descending, takes the largest gap between neighbors and puts
    ``omega_term`` at its midpoint; the clusters above it are the terminals.

    Parameters
    ----------
    term_values : array-like of shape (n_clusters,)
        Terminality scores, e.g. the ``terminality`` column of :func:`cluster_terminality`.

    Returns
    -------
    omega : float
        Suggested ``omega_term``.
    info : dict
        ``clear`` (bool): the largest gap is >= 2x the next-largest, so any omega
        inside it gives the same terminals. Also ``gap``/``gap_lo``/``gap_hi``,
        ``n_above`` (clusters above the cut), and ``next_gap``.
    """
    v = np.sort(np.asarray(term_values, float))[::-1]
    if v.size == 0:
        return 0.5, dict(clear=False, gap=0.0, gap_lo=0.0, gap_hi=1.0, n_above=0)
    if v.size == 1:
        return float(v[0]) - 1e-6, dict(clear=True, gap=float("inf"),
                                        gap_lo=0.0, gap_hi=float(v[0]), n_above=1)
    gaps = v[:-1] - v[1:]
    k = int(np.argmax(gaps))
    gap, lo, hi = float(gaps[k]), float(v[k + 1]), float(v[k])
    others = np.delete(gaps, k)
    next_gap = float(others.max()) if others.size else 0.0
    return (lo + hi) / 2.0, dict(
        clear=gap >= 2.0 * max(next_gap, 1e-9),
        gap=gap, gap_lo=lo, gap_hi=hi, n_above=k + 1, next_gap=next_gap,
    )


def top_terminals(term, n_lineages):
    """Cut the terminality ranking by count: keep its top ``n_lineages`` clusters,
    in the canonical order (see :func:`_rank_order`).

    Parameters
    ----------
    term : pandas.DataFrame
        Ranking from :func:`cluster_terminality`.
    n_lineages : int
        How many clusters to keep.

    Returns
    -------
    names : list of str
        The chosen terminal clusters, most terminal first.
    tied : list of str
        Clusters sharing the terminality score that straddles the cut, i.e. split
        arbitrarily rather than by the flow. Empty if the cut falls cleanly.
    """
    g = _rank_order(term)
    names = [str(i) for i in g.index[:n_lineages]]
    v = g["term_score"].to_numpy(float)
    tied = []
    if 0 < n_lineages < v.size and v[n_lineages - 1] == v[n_lineages]:
        tied = [str(i) for i in g.index[v == v[n_lineages]]]
    return names, tied


def terminal_cells(adata, chain, cluster_key, terminal_clusters, q_quantile=0.0):
    """Collect the cells of each terminal cluster that the flow actually reaches.

    One lineage per cluster. Raising ``q_quantile`` keeps only the cells most likely to
    stop there, i.e. those with the highest absorption probability q within the cluster.

    Parameters
    ----------
    adata : anndata.AnnData
        Object with the cluster labels in ``obs[cluster_key]``.
    chain : dict
        Output of :func:`build_chain`; ``live`` and ``q`` are used.
    cluster_key : str
        ``obs`` column with the cluster labels.
    terminal_clusters : sequence of str
        Labels to treat as terminal states.
    q_quantile : float, default 0.0
        Drop cells below this quantile of q within their cluster. 0 keeps every live cell.

    Returns
    -------
    dict of {str: ndarray}
        Cluster label -> positional indices of its terminal cells, ready to hand to
        :func:`absorption`.
    """
    lab = adata.obs[cluster_key].astype(str).to_numpy()
    out = {}
    for c in terminal_clusters:
        m = (lab == c) & chain["live"]
        if q_quantile > 0:
            thr = np.quantile(chain["q"][m], q_quantile)
            m &= chain["q"] >= thr
        out[c] = np.where(m)[0]
    return out


# ---------------------------------------------------------------------------
# Cluster-level trajectory  (the PAGA-free cluster graph)
# ---------------------------------------------------------------------------
def _flow_cluster_names(adata, cluster_key):
    """Cluster labels present, natural-sorted (``"10"`` after ``"9"``) for a stable,
    human-readable order; cosmetic only, cluster_flux_matrix builds its own index."""
    from natsort import natsorted
    labs = {str(c) for c in adata.obs[cluster_key].astype(str).to_numpy()}
    return [str(c) for c in natsorted(labs)]


def cluster_flux_matrix(adata, cluster_key, cluster_names, flow_key="dt_flow",
                        edge_threshold=1e-6):
    """Pivot the cross-cluster flux into the directed cluster-by-cluster matrix F.

    ``F[i, j]`` is the flow summed over every active edge ``u -> v`` with ``lab[u]`` the
    i-th name and ``lab[v]`` the j-th. J is non-negative and both orientations of an
    undirected pair are separate rows, so direction lives ONLY in the ``i, j`` / ``j, i``
    asymmetry and never in the sign of J.

    Parameters
    ----------
    adata : anndata.AnnData
        Object holding the flow in ``uns[flow_key]`` and labels in ``obs[cluster_key]``.
    cluster_key : str
        ``obs`` column with the cluster labels.
    cluster_names : sequence of str
        Clusters to score, one row and column each, in the order given. Labels
        outside this list are ignored.
    flow_key : str, default "dt_flow"
        ``uns`` key of the QROT flow.
    edge_threshold : float, default 1e-6
        Minimum flow, as a fraction of the transported mass (supplies sum to 1), for an edge to count.

    Returns
    -------
    F : ndarray of shape (K, K)
        Gross directed flux between clusters; diagonal is 0 (self-loops dropped).
    index : dict of {str: int}
        Label to row/column position.
    """
    names = [str(c) for c in cluster_names]
    index = {c: i for i, c in enumerate(names)}
    K = len(names)
    df = _cross_cluster_flux(adata, cluster_key, flow_key=flow_key,
                             edge_threshold=edge_threshold)
    d = df[(df["src"].isin(index) & df["dst"].isin(index)).to_numpy()]
    ci = d["src"].map(index).to_numpy(np.int64)
    cj = d["dst"].map(index).to_numpy(np.int64)
    # astype: bincount of no edges comes back int64
    F = np.bincount(ci * K + cj, weights=d["j"].to_numpy(float),
                    minlength=K * K).reshape(K, K).astype(float, copy=False)
    return F, index


def net_flux_matrix(adata, cluster_key, cluster_names, flow_key="dt_flow",
                    edge_threshold=1e-6):
    """Reduce the gross flux to the antisymmetric NET flux and its total transported mass.

    ``NET = F - F.T``, so an A/B pair that merely churns mass back and forth cancels and
    only the net transport survives. Cluster-level conservation makes
    ``sum_j NET[i, j]`` the net supply of cluster i, so ``M = 0.5 * sum_i |sum_j NET[i, j]|``
    is the total mass the flow carries from source clusters to sink clusters. That is what
    makes ``theta`` a fraction of transported mass rather than an arbitrary cutoff.

    Parameters
    ----------
    adata : anndata.AnnData
        Object holding the flow and the cluster labels.
    cluster_key : str
        ``obs`` column with the cluster labels.
    cluster_names : sequence of str
        Clusters to score, one row and column each.
    flow_key : str, default "dt_flow"
        ``uns`` key of the QROT flow.
    edge_threshold : float, default 1e-6
        Minimum flow, as a fraction of the transported mass (supplies sum to 1), for an edge to count.

    Returns
    -------
    NET : ndarray of shape (K, K)
        Net flux, antisymmetric, entries below ``EPS`` in magnitude zeroed.
    M : float
        Total transported mass.
    index : dict of {str: int}
        Label to row/column position.
    fallback : bool
        True when conservation didn't hold well enough to read M off the row sums,
        so ``max |NET|`` was used instead (theta then a fraction of the largest edge,
        not of transported mass). The caller must report it.
    """
    F, index = cluster_flux_matrix(adata, cluster_key, cluster_names, flow_key=flow_key,
                                   edge_threshold=edge_threshold)
    NET = F - F.T
    NET[np.abs(NET) < EPS] = 0.0
    M = 0.5 * float(np.abs(NET.sum(axis=1)).sum())
    fallback = False
    if (not np.isfinite(M)) or M <= EPS:
        # Conservation broken (e.g. a flow whose supply does not balance): say so loudly.
        M = float(np.abs(NET).max()) if NET.size else 0.0
        fallback = True
    return NET, M, index, fallback


def _median_cluster_pt(adata, cluster_key, cluster_names, pt_key):
    """Median (not mean) pseudotime per cluster; median avoids off-LCC reconnection
    dragging a mean toward borrowed values. NaN where pt_key is absent or the
    cluster holds no finite value."""
    import pandas as pd
    if pt_key not in adata.obs:
        return np.full(len(cluster_names), np.nan)
    lab = adata.obs[cluster_key].astype(str).to_numpy()
    v = adata.obs[pt_key].to_numpy(float)
    med = pd.Series(np.where(np.isfinite(v), v, np.nan)).groupby(lab).median()
    # + 0.0 turns pandas' -0.0 into the +0.0 np.median gives
    return med.reindex([str(c) for c in cluster_names]).to_numpy(float) + 0.0


def _meg(K, pairs, NET):
    """Minimum equivalent graph: drop every edge another path already implies.

    On a DAG this is nx.transitive_reduction, which is unique and equals the minimum
    equivalent graph (Aho, Garey & Ullman 1972). Off one, condense the strongly
    connected components, reduce the resulting DAG (each of its edges represented by
    the heaviest candidate between the two components), and keep one directed cycle
    inside each component, the first found by depth-first search. When that cycle
    passes through every node of its component the result is again a minimum
    equivalent graph (Moyles & Thompson 1969); otherwise it is a heuristic, since the
    problem is NP-hard inside a strongly connected component.

    Deterministic: condensation, then transitive reduction, then one DFS cycle per
    strongly connected component.
    """
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(range(K))
    G.add_edges_from(pairs)
    if nx.is_directed_acyclic_graph(G):
        return set(nx.transitive_reduction(G).edges())

    C = nx.condensation(G)
    memb = C.graph["mapping"]
    out = []
    for a, b in nx.transitive_reduction(C).edges():
        cross = [(u, v) for u, v in pairs if memb[u] == a and memb[v] == b]
        out.append(max(cross, key=lambda e: float(NET[e])))
    for scc in nx.strongly_connected_components(G):
        if len(scc) < 2:
            continue
        sub = G.subgraph(scc)
        try:
            out += [(u, v) for u, v, *_ in
                    nx.find_cycle(sub, orientation="original")][:len(scc)]
        except Exception:
            out += list(sub.edges())
    return set(out)


def cluster_graph(adata, cluster_key, flow_key="dt_flow", pt_key="pseudotime",
                  edge_rule="meg", theta=0.05, term_tol=0.40, simplify=True,
                  edge_threshold=1e-6, enforce_acyclic=True, repair=True):
    """Abstract the cell-level flow into a directed graph over clusters.

    One node per cluster; every ordered pair with positive net flux is a candidate
    edge, oriented by the sign of NET and never by a pseudotime comparison. The
    candidates are then reduced in this order:

    1. leak prune (``term_tol``): a cluster retaining most of the net flux arriving at
       it is a dead end, so its outgoing candidates are dropped wherever the target
       keeps another incoming one;
    2. simplification (``simplify``): the minimum equivalent graph (``edge_rule="meg"``,
       every edge another path already implies is dropped) or a mass cutoff
       (``edge_rule="theta"``); skipped entirely when ``simplify=False``;
    3. repair (``repair``): dropped candidates that merge components are re-admitted,
       heaviest first, so the graph cannot fall apart; leak-pruned edges stay out;
    4. acyclicity guard (``enforce_acyclic``): edges enter heaviest first and any that
       would close a directed cycle is reversed (counted in ``n_cycle_flips``). Summing
       flow over a cluster boundary can create cycles the cell-level DAG does not have.
       Pass ``enforce_acyclic=False`` for a genuinely cyclic population, where a cycle
       is the result, not an error.

    See :func:`deltatrace.tl.cluster_trajectory` for more.

    Parameters
    ----------
    adata : anndata.AnnData
        Object holding the flow in ``uns[flow_key]`` and labels in ``obs[cluster_key]``.
    cluster_key : str
        ``obs`` column with the cluster labels.
    flow_key : str, default "dt_flow"
        ``uns`` key of the QROT flow.
    pt_key : str, default "pseudotime"
        ``obs`` column with the pseudotime, used to color nodes and to run the
        median-inversion detector. An absent column disables both rather than raising.
    edge_rule : {"meg", "theta"}, default "meg"
        Simplification rule; read only when ``simplify`` is True.
    theta : float, default 0.05
        Under ``edge_rule="theta"``, keep ``i -> j`` when ``NET[i, j] >= theta * M``.
        Validated by the caller; ignored under ``"meg"``.
    term_tol : float or None, default 0.40
        Leak-prune ratio; ``None`` disables the prune.
    simplify : bool, default True
        Run step 2. ``False`` keeps every candidate that survives the leak prune.
    edge_threshold : float, default 1e-6
        Minimum flow, as a fraction of the transported mass (supplies sum to 1), for an edge to count.
    enforce_acyclic : bool, default True
        Break cluster-level cycles by reversing the closing edge. ``False`` keeps every
        kept edge in its raw NET-sign direction, so a real cycle in the flow stays a
        cycle in the graph; ``n_cycle_flips``/``n_cycle_unresolved`` are then always 0.
    repair : bool, default True
        Re-admit dropped candidates that merge components, so simplification cannot
        disconnect the graph and orphan a cluster's cells. A no-op after a minimum
        equivalent graph of acyclic candidates, which preserves reachability. Counted
        in ``n_repair_edges_added``.

    Returns
    -------
    dict
        See :func:`deltatrace.tl.cluster_trajectory`, which returns this unchanged,
        for the key-by-key description.
    """
    import networkx as nx

    df = _cross_cluster_flux(adata, cluster_key, flow_key=flow_key,
                             edge_threshold=edge_threshold)
    carriers = set(df["src"].unique()) | set(df["dst"].unique())
    if len(carriers) < 2:
        raise ValueError(
            f"Only {len(carriers)} cluster(s) carry cross-cluster flux "
            f"({sorted(carriers)}): cluster at a finer resolution, or lower "
            f"edge_threshold={edge_threshold:g}."
        )

    names = _flow_cluster_names(adata, cluster_key)
    K = len(names)
    NET, M, _index, fallback = net_flux_matrix(adata, cluster_key, names,
                                               flow_key=flow_key,
                                               edge_threshold=edge_threshold)
    cluster_pt = _median_cluster_pt(adata, cluster_key, names, pt_key)

    # Candidate edges: every ordered pair with net transport. The leak prune walks them
    # in cluster-index order (the order matters only when every incoming candidate of a
    # cluster comes from leaky clusters); the repair step walks them heaviest first.
    pairs = [(i, j) for i in range(K) for j in range(K) if NET[i, j] > 0.0]
    if not pairs:
        raise ValueError(
            f"No net flow between any two clusters of obs['{cluster_key}']: the flow is "
            "empty or stale, so rerun tl.qrot."
        )
    cand = sorted(((i, j, float(NET[i, j])) for i, j in pairs),
                  key=lambda t: (-t[2], t[0], t[1]))

    if edge_rule not in ("meg", "theta"):
        raise ValueError(f"edge_rule must be 'meg' or 'theta', got {edge_rule!r}")

    # Leak prune. A cluster absorbing far more than it emits is terminal, so mass apparently
    # leaving it is leakage. Never orphans dst: it must keep another incoming candidate.
    n_leak = 0
    if term_tol is not None:
        pos = np.maximum(NET, 0.0)
        ratio = pos.sum(axis=1) / np.clip(pos.sum(axis=0), EPS, None)
        indeg = {}
        for _i, j in pairs:
            indeg[j] = indeg.get(j, 0) + 1
        keep = []
        for i, j in pairs:
            if ratio[i] < float(term_tol) and indeg.get(j, 0) >= 2:
                indeg[j] -= 1
                n_leak += 1
            else:
                keep.append((i, j))
        pairs = keep

    # Edge selection (skipped when simplify=False: prune only).
    if not simplify:
        kept = set(pairs)
    elif edge_rule == "meg":
        kept = _meg(K, pairs, NET)
    else:
        # >= so an edge ON the cut is KEPT.
        cut = float(theta) * M
        kept = {(i, j) for i, j in pairs if float(NET[i, j]) >= cut}
    n_pruned = len(cand) - len(kept)

    # Re-admit the heaviest dropped edges that merge components, so pruning cannot
    # disconnect the graph and orphan a cluster's cells. Leak-pruned edges stay out.
    n_repair = 0
    if repair:
        parent = list(range(K))

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a, b):
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True

        for i, j in kept:
            _union(i, j)
        survivors = set(pairs)
        for i, j, _w in cand:                     # already mass-descending
            if (i, j) in kept or (i, j) not in survivors:
                continue
            if _union(i, j):
                kept.add((i, j))
                n_repair += 1

    # Clusters no net transport touches: left isolated, never attached by a non-flow signal.
    n_zero_flux = int((np.abs(NET).sum(axis=1) <= 0.0).sum())

    # DAG GUARD: cluster-level acyclicity is imposed, not free (see the docstring) --
    # unless enforce_acyclic=False, which keeps every edge as NET actually orients it.
    order = sorted(kept, key=lambda ij: (-abs(float(NET[ij[0], ij[1]])), ij[0], ij[1]))
    Gd = nx.DiGraph()
    Gd.add_nodes_from(range(K))
    if enforce_acyclic:
        final, n_flipped, n_unresolved = [], 0, 0
        for (src, dst) in order:
            if not nx.has_path(Gd, dst, src):
                Gd.add_edge(src, dst); final.append((src, dst, False))
            elif not nx.has_path(Gd, src, dst):
                Gd.add_edge(dst, src); final.append((dst, src, True)); n_flipped += 1
            else:
                Gd.add_edge(src, dst); final.append((src, dst, False)); n_unresolved += 1
    else:
        final = [(src, dst, False) for src, dst in order]
        Gd.add_edges_from(order)
        n_flipped = n_unresolved = 0
    n_components = nx.number_weakly_connected_components(Gd)

    # DAG-RISK DETECTOR, on the PRE-guard orientation, so it checks the guard, not itself.
    n_inverted = -1
    if np.isfinite(cluster_pt).all():
        n_inverted = int(sum(1 for (i, j) in kept if cluster_pt[i] > cluster_pt[j]))

    import pandas as pd
    # final is already in the guard's insertion order: heaviest first, then by position.
    rows = [dict(src=names[a], dst=names[b], i=a, j=b,
                 weight=abs(float(NET[a, b])), frac=abs(float(NET[a, b])) / max(M, EPS),
                 flipped=bool(fl))
            for a, b, fl in final]
    edges = pd.DataFrame(rows, columns=["src", "dst", "i", "j", "weight", "frac",
                                        "flipped"])

    pos = np.maximum(NET, 0.0)
    out_flux = pos.sum(axis=1)
    in_flux = pos.sum(axis=0)
    return dict(
        names=names,
        cluster_key=str(cluster_key),
        pt_key=str(pt_key),
        edge_threshold=float(edge_threshold),
        net_flux=NET,
        edges=edges,
        throughput=np.maximum(in_flux, out_flux),
        in_flux=in_flux,
        out_flux=out_flux,
        cluster_pt=cluster_pt,
        M=float(M),
        theta=float(theta),
        simplify=bool(simplify),
        enforce_acyclic=bool(enforce_acyclic),
        n_cycle_flips=int(n_flipped),
        n_cycle_unresolved=int(n_unresolved),
        n_median_inverted=int(n_inverted),
        n_candidate_pairs=len(cand),
        n_pruned_by_theta=int(n_pruned), n_leak_pruned=int(n_leak),
        n_repair_edges_added=int(n_repair),
        n_zero_flux_clusters=int(n_zero_flux),
        n_components=int(n_components),
        M_fallback=bool(fallback),
    )


# ---------------------------------------------------------------------------
# Path objects  (the widest route through a lineage's flow)
# ---------------------------------------------------------------------------
def _widest_path(chain, JL_edges, JL, src_idx, dst_idx):
    """Trace the heaviest progenitor -> terminal route through a lineage's flow.

    Max-bottleneck (widest) path: of all routes it maximizes the smallest edge flow along
    the way, so THIS is 'the most mass' path. It is not a Viterbi score, which multiplies
    edge weights and so penalizes a long route for its length alone.

    Parameters
    ----------
    chain : dict
        Only ``chain["n"]``, the number of cells, is read.
    JL_edges : ndarray of shape (m, 2)
        Edges as (from cell, to cell) pairs.
    JL : ndarray of shape (m,)
        Flow on each edge, e.g. the lineage flow J^L.
    src_idx : array-like of int
        Cells the route may start from.
    dst_idx : array-like of int
        Terminal cells the route may end at; the search stops at the first one it reaches,
        which is the one with the widest bottleneck.

    Returns
    -------
    path : list of int
        Cells from source to terminal, empty if no route carries flow.
    capacity : float
        Flow of the route's narrowest edge, 0.0 if there is no route.
    """
    import heapq
    n = chain["n"]
    # out-edges of cell v are nbr/wt[ptr[v]:ptr[v + 1]], in input order
    k = JL > EPS
    a = JL_edges[k, 0]
    o = np.argsort(a, kind="stable")
    nbr, wt = JL_edges[k, 1][o].tolist(), JL[k][o].tolist()
    ptr = np.searchsorted(a[o], np.arange(n + 1)).tolist()
    cap = np.zeros(n); prev = np.full(n, -1)
    dst = set(np.asarray(dst_idx).tolist())
    h = []
    for s in np.asarray(src_idx):
        cap[s] = np.inf; heapq.heappush(h, (-np.inf, int(s)))
    seen = np.zeros(n, bool); best = -1
    while h:
        c, v = heapq.heappop(h); c = -c
        if seen[v]:
            continue
        seen[v] = True
        if v in dst:
            best = v; break
        for i in range(ptr[v], ptr[v + 1]):
            u = nbr[i]; nc = min(c, wt[i])
            if nc > cap[u] + EPS:
                cap[u] = nc; prev[u] = v; heapq.heappush(h, (-nc, u))
    if best < 0:
        return [], 0.0
    path = [best]
    while prev[path[-1]] >= 0:
        path.append(int(prev[path[-1]]))
    return path[::-1], float(cap[best])


# ---------------------------------------------------------------------------
# Write  (the fate keys written into adata)
# ---------------------------------------------------------------------------
def lineages_from_chain(adata, source_idx, terminal_clusters, cluster_key,
                        flow_key="dt_flow", pt_key="pseudotime",
                        edge_threshold=0, suffix=""):
    """Assign every cell a lineage and a confidence, and write them into ``adata``.

    Runs build_chain -> terminal_cells -> absorption, then writes the fate keys
    (``obsm['path_cellweights']``, ``obsm['path_pseudotime']``,
    ``obsm['path_flowmass']``, ``obs['lineage']``, ``obs['lineage_confidence']``,
    ``obs['fate_leakage']``, and one ``obs['fate_<lineage>']`` column per lineage --
    the columns of ``path_cellweights`` where scanpy can color by them, with ``/`` in a
    lineage name written as ``_``); see those functions for the maths.

    Parameters
    ----------
    adata : anndata.AnnData
        Modified in place.
    source_idx : array-like of int
        Positions of the progenitor cells the flow is emitted from.
    terminal_clusters : sequence of str
        Cluster labels to use as terminal states, one lineage each.
    cluster_key : str
        ``obs`` column with the cluster labels.
    flow_key, pt_key : str
        ``uns`` / ``obs`` keys of the QROT flow and pseudotime.
    edge_threshold : float, default 0
        Minimum flow for an edge to count; 0 keeps exactly the edges the optimal flow uses.
    suffix : str, default ""
        Appended to every written key, so several runs can live side by side.

    Returns
    -------
    dict
        ``chain`` plus ``A``, ``b``, ``F`` (see :func:`absorption`) and ``names``,
        the lineage names in column order.
    """
    chain = build_chain(adata, source_idx, flow_key=flow_key, edge_threshold=edge_threshold)
    groups = terminal_cells(adata, chain, cluster_key, terminal_clusters)
    groups = {str(k): v for k, v in groups.items() if len(v)}
    if not groups:
        raise ValueError("dt_chain: no terminal cells found for "
                         f"{terminal_clusters} in obs['{cluster_key}']")
    names = list(groups)
    if "leakage" in names:
        raise ValueError(
            "a terminal cluster named 'leakage' would collide with obs['fate_leakage']; "
            "rename the cluster."
        )
    A, b, F = absorption(chain, groups)
    W = np.nan_to_num(F, nan=0.0)  # off-path rows -> 0

    pt = adata.obs[pt_key].to_numpy(float)
    on_path = W.sum(1) > EPS
    adata.obsm[f"path_cellweights{suffix}"] = W
    adata.obsm[f"path_flowmass{suffix}"] = np.nan_to_num(A, nan=0.0)
    adata.obsm[f"path_pseudotime{suffix}"] = np.tile(pt[:, None], (1, W.shape[1]))
    adata.uns[f"path_lineages{suffix}"] = names
    adata.obs[f"lineage{suffix}"] = np.where(on_path, np.array(names, object)[W.argmax(1)],
                                    "unassigned")
    adata.obs[f"lineage_confidence{suffix}"] = np.where(on_path, W.max(1), np.nan)
    adata.obs[f"fate_leakage{suffix}"] = np.where(on_path, 1.0 - b, np.nan)
    # Per-fate absorption probability, NaN off-path like lineage_confidence. h5 stores
    # are dropping support for "/" in keys, so a cluster like "L5/6" is written as
    # fate_L5_6 (its name in `names` and obs['lineage'] is untouched).
    fate_cols = [f"fate_{name.replace('/', '_')}{suffix}" for name in names]
    if len(set(fate_cols)) < len(fate_cols):
        raise ValueError(f"terminal clusters {names} collide once '/' is replaced by '_' "
                         "in their obs['fate_<lineage>'] column names; rename one.")
    for k, col in enumerate(fate_cols):
        adata.obs[col] = np.where(on_path, W[:, k], np.nan)

    # fate weights must be a proper distribution wherever the flow arrives
    assert np.allclose(W[on_path].sum(1), 1.0, atol=1e-6), \
        "path_cellweights rows must sum to 1 for on-path cells"

    if _log.wants_detail():
        n_on = int(on_path.sum())
        _log.detail(
            f"lineages {names} | {n_on}/{adata.n_obs} on-path | "
            f"mean confidence {np.nanmean(adata.obs[f'lineage_confidence{suffix}']):.3f} | "
            f"mean leakage {np.nanmean(adata.obs[f'fate_leakage{suffix}']):.3f}"
        )
    return dict(chain=chain, A=A, b=b, F=F, names=names)


# ---------------------------------------------------------------------------
# tl.cluster_trajectory
# ---------------------------------------------------------------------------
@_log.stage("abstracting the flow into a cluster graph")
def cluster_trajectory(
    adata,
    cluster_key: str = "clusters",
    *,
    flow_key: str | None = None,
    edge_rule: str = "meg",
    theta: float = 0.05,
    term_tol: float | None = 0.40,
    simplify: bool = True,
    edge_threshold: float = 1e-6,
    enforce_acyclic: bool = True,
    repair: bool = True,
    pt_key: str | None = None,
    key_added: str | None = None,
    verbose: int | bool = 1,
) -> dict:
    """Abstract the cell-level flow into a directed graph over clusters.

    One node per cluster, a candidate edge wherever net mass crosses a cluster boundary:

        F[i,j] = sum of J over active edges from cluster i to cluster j
        NET    = F - F.T                       net flux, antisymmetric
        M      = 0.5 * sum_i |sum_j NET[i,j]|  total transported mass
        candidate i -> j  iff  NET[i,j] > 0

    Edges are oriented by the sign of NET, not by comparing pseudotimes, so the
    arrows are the transport itself. The candidates are then reduced in four steps:

    1. leak prune: a cluster with ``out_flux/in_flux < term_tol`` retains most of what
       reaches it, so its outgoing candidates are spillover and are dropped wherever
       the target keeps another incoming candidate;
    2. simplification: with ``simplify=True`` (default) the minimum equivalent graph of
       the survivors (``edge_rule="meg"``: every edge another path already implies is
       dropped, however much flux it carries) or the mass cutoff
       ``NET[i,j] >= theta * M`` (``edge_rule="theta"``); ``simplify=False`` stops after
       the prune and keeps every surviving candidate;
    3. repair: dropped candidates that merge components are re-admitted heaviest first;
    4. acyclicity guard: summing flow over a cluster boundary can introduce cycles even
       though the cell-level flow is a DAG, so edges enter heaviest first and any that
       would close one is reversed (counted in ``n_cycle_flips``). Pass
       ``enforce_acyclic=False`` for a genuinely cyclic population, so a real cycle is
       kept as one instead of broken.

    Parameters
    ----------
    adata : anndata.AnnData
        Object QROT wrote into, holding ``uns[flow_key]``, the labels in
        ``obs[cluster_key]`` and (optionally) ``obs[pt_key]``.
    cluster_key : str, default "clusters"
        ``obs`` column holding the cluster labels.
    flow_key : str, optional
        ``uns`` key of the flow written by :func:`qrot`. Overrides ``key_added``
        when given.
    edge_rule : {"meg", "theta"}, default "meg"
        Simplification rule (step 2); read only when ``simplify`` is True.
    theta : float, default 0.05
        Under ``edge_rule="theta"``, the edge cutoff as a fraction of the transported
        mass M, strictly inside ``(0, 1)``. Ignored under ``"meg"``.
    term_tol : float or None, default 0.40
        Leak-prune ratio (step 1). ``None`` disables the prune.
    simplify : bool, default True
        Run step 2. ``False`` gives the leak prune alone, followed by steps 3 and 4.
    edge_threshold : float, default 1e-6
        Minimum flow, as a fraction of the transported mass (supplies sum to 1), for an edge to count.
    enforce_acyclic : bool, default True
        Break cluster-level cycles by reversing the closing edge. Set ``False`` to
        keep every kept edge in its raw NET-sign direction instead --
        ``n_cycle_flips``/``n_cycle_unresolved`` are then always 0.
    repair : bool, default True
        Re-admit dropped candidates that merge components, so simplification cannot
        disconnect the graph and orphan a cluster's cells. Under ``"theta"`` the cutoff
        is then no longer literal. Counted in ``n_repair_edges_added``.
    pt_key : str, optional
        ``obs`` column with the pseudotime, used only to color nodes and run the
        median-inversion check, never to orient an edge. Missing pseudotime
        disables both rather than raising.
    key_added : str, optional
        Namespace of the run, matching the one given to :func:`qrot`. Result is
        written to ``uns['dt_cluster_trajectory<_key_added>']``.
    verbose : int or bool, default 1
        Logging verbosity.

    Returns
    -------
    dict
        Also written to ``uns['dt_cluster_trajectory<_key_added>']``:

        - ``"names"``: clusters, sorted; indexes every array below.
        - ``"net_flux"``: ``(K, K)`` NET matrix.
        - ``"edges"``: retained directed edges as a DataFrame (``src``/``dst``,
          ``i``/``j``, ``weight``, ``frac``, ``flipped``).
        - ``"throughput"``, ``"in_flux"``, ``"out_flux"``: ``(K,)`` per-cluster flux.
        - ``"cluster_pt"``: ``(K,)`` median pseudotime; NaN where undefined.
        - ``"M"``, ``"theta"``, ``"simplify"``, ``"enforce_acyclic"``: transported
          mass, cutoff, and whether simplification and the acyclicity guard ran.
        - ``"n_cycle_flips"``, ``"n_cycle_unresolved"``: acyclicity guard counters.
        - ``"n_median_inverted"``: retained edges running backwards in the raw NET
          orientation before the guard ran; ``-1`` if no cluster had a finite median.
        - ``"n_candidate_pairs"``, ``"n_leak_pruned"``, ``"n_pruned_by_theta"``
          (candidates dropped by steps 1 and 2 together), ``"n_repair_edges_added"``,
          ``"n_zero_flux_clusters"``, ``"n_components"``, ``"M_fallback"``:
          candidate/pruned/re-admitted pair counts, zero-flux cluster count,
          connected components, and whether M fell back off the conservation law.
    """

    _sfx = _suffix(key_added)
    flow_key = flow_key or f"dt_flow{_sfx}"
    pt_key = pt_key or f"pseudotime{_sfx}"

    if edge_rule not in ("meg", "theta"):
        raise ValueError(f"edge_rule must be 'meg' or 'theta', got {edge_rule!r}")
    if edge_rule == "theta" and (
            isinstance(theta, bool)
            or not isinstance(theta, (int, float, np.integer, np.floating))
            or not 0.0 < float(theta) < 1.0):
        raise ValueError(f"theta must be a number strictly inside (0, 1), got {theta!r}.")
    theta_f = float(theta)
    if adata.uns.get(flow_key) is None:
        raise ValueError(
            f"No flow data in adata.uns['{flow_key}']. Run qrot() with adata_basis set "
            f"(and the same key_added) first."
        )

    res = cluster_graph(adata, cluster_key, flow_key=flow_key, pt_key=pt_key,
                               edge_rule=edge_rule, theta=theta_f, term_tol=term_tol,
                               simplify=simplify, edge_threshold=edge_threshold,
                               enforce_acyclic=enforce_acyclic, repair=repair)
    adata.uns[f"dt_cluster_trajectory{_sfx}"] = res

    if _log.wants_detail():
        rule = edge_rule if simplify else "prune only"
        _log.detail(
            f"{len(res['names'])} clusters, M={res['M']:.4f} | "
            f"{res['n_candidate_pairs']} candidate pairs -> {len(res['edges'])} edges "
            f"({rule}, leak-pruned {res['n_leak_pruned']}, "
            f"dropped {res['n_pruned_by_theta']} in all, "
            f"repair +{res['n_repair_edges_added']}, "
            f"zero-flux {res['n_zero_flux_clusters']}, "
            f"{res['n_components']} component(s))"
        )
        if enforce_acyclic:
            _log.detail(
                f"acyclicity guard: {res['n_cycle_flips']} flip(s), "
                f"{res['n_cycle_unresolved']} unresolved, "
                f"{res['n_median_inverted']} median-inverted edge(s)"
            )
        else:
            _log.detail("acyclicity guard: disabled, cycles may be present")
        if res["M_fallback"]:
            _log.detail("M fell back to max|NET|, so theta is not a fraction of the transported mass")
    return res
