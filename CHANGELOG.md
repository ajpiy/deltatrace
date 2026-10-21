# Changelog

## 0.3.0

### Changed

- `pp.integrate`: consolidated preprocessing into `dt.pp.integrate` as the unified graph integration entry point.
- `tl.endpoints`: cleaned parameter signatures and removed legacy `_renamed_kwargs` wrapper and `blocked_flows` argument.
- `tl.qrot` and `tl.trajectory` default to `tau="auto"`: 0.3 times the cells in the largest connected piece
  of the graph, the rule selected on the Figure 2 benchmark (it replaces the fixed 100, under which the flow
  concentrated on few edges in large graphs). Pass `tau=100` for the previous results. `tl.qrot_graph` keeps
  100 as its default and also accepts `"auto"`. The tau solved at is stored in `uns["dt_flow"]["tau"]` and
  returned by `qrot_graph` as `"tau"`.
- `tl.qrot` / `tl.qrot_graph`: each Newton step's linear system is factored by SuperLU unless its factors
  would be dense, which is counted exactly before factoring (in SuperLU's own ordering); then it is solved by
  conjugate gradients preconditioned with algebraic multigrid (PyAMG, deterministic). Same flows (to
  rounding) and KKT conditions. On single-cell kNN graphs at the new tau: 61,819 cells 346 s -> 15 s and
  2.1 -> 0.5 GB; 30,000 cells 5-8x faster; unchanged where the factors stay sparse (small graphs,
  tau 100, low-dimensional graphs).
- `pyamg` is listed as a dependency (MIT).
- `tl.qrot` / `tl.qrot_graph`: custom Numba-compiled Steihaug--Toint Newton-CG trust-region warm start capped at 100
  steps with TRON-style inner CG capping (`max_cg=20`), relative dual plateau exit (`ftol=1e-6`), and two-tier
  radius expansion, running 2-8x faster than SciPy's `minimize` with zero Python callback overhead. Newton
  runs up to 3000 steps per tau level (was 300), and the Newton line search and per-step residual are compiled
  with numba.
- `numba` is listed as a dependency (scanpy already requires it).

### Fixed

- `tl.cluster_trajectory` / `cluster_flux_matrix`: fixed integer overflow bug when cluster count $K \ge 12$
  where pandas `Categorical.codes` defaulted to signed `int8`, causing negative indexing errors in `np.bincount`.
- `tl.qrot` no longer fails with "did not converge" on graphs with many disconnected islands or with all costs
  at the `cost_range` floor and a tiny `tau`: those need more than the old 300 Newton steps per level.
- `tl.qrot_graph`'s `tau` docstring no longer suggests rescaling `tau` by the total mass.

## 0.2.0

### Added

- `tl.qrot_graph`: the QROT solver on any graph in SciPy's csgraph form, with source and sink masses `a` and `b`.

### Changed

- `tl.qrot` solves the transport dual with SciPy instead of CVXPY + CLARABEL: 10-30x faster, 4-15x less
  memory, exact KKT conditions. `cvxpy`, `clarabel` and `solver_verbose` are gone.
- Cells without flow sit at the midpoint of their optimal potential range; `pseudotime_raw` is NaN in graph
  pieces without a source or sink.
- Defaults `p_keep` 0.32 -> 0.3 and `p_end` 0.10 -> 0.15, the Figure 2 benchmark's selection (geo4).
- Default `edge_threshold` 1e-6 -> 0: lineages use every edge the flow carries (cluster graphs keep 1e-6).
- Default velocity-grid `k` 10 -> 25, so streamlines stay continuous on the sparse exact flow.
- `pp.wasserstein` returns condensed distances (`uns["distances"]`, SciPy `pdist` order); `pp.integrate` takes
  them or a square matrix, at a fraction of the memory.
- Terminality columns `inf` / `out` / `W` / `Term` are now `flow_in` / `flow_out` / `absorbed` / `term_score`.
- Errors and warnings are one line: what went wrong and how to fix it.
- Library routines replace hand-rolled loops: bit-identical results in less time and memory.

### Fixed

- `tl.qrot` no longer stalls on whole-number costs or uniform demand; it checks `tau` and `cost_range` and
  warns when the flow misses the demand.
- `tl.lineages` solves absorption exactly, so a flow cycle no longer drops mass, and zeroes fate shares
  below 1e-10 (rounding).
- A pair at distance 0 keeps its edge, so a dense matrix must not use 0 for unknown: use NaN, or leave the
  pair out of a sparse matrix.
- `pp.wasserstein` sets pairs that hit the EMD iteration cap to NaN and warns on a nearly uniform mapping.
- Clear errors for repeated `obs_names`, a wrong-length demand, a tiny negative EMD, more lineages than
  palette colors, and NaN coordinates in `pl.cluster_trajectory`.
- `pl.pseudotime`'s colorbar mappable is always `ax.collections[0]`; `pl.cluster_trajectory` no longer makes
  matplotlib log "Ignoring fixed x limits".

## 0.1.0

First public release.

- `pp`: `wasserstein`, `integrate`, `graphs`
- `tl`: `trajectory` (endpoints -> qrot -> velocity grid -> lineages), `terminality`,
  `widest_path`, `cluster_trajectory`
- `pl`: `pseudotime`, `velocity`, `endpoints`, `lineages`, `cluster_trajectory`;
  every result is a plain `obs` column, so scanpy can plot it too
- `datasets`: `crypt`, a crypt-shaped example tissue whose two walls run the same
  expression program (the former line `toy` now lives only in the test suite)
- Defaults are the benchmarked configuration for the intersection graph.
- Python >= 3.12; tested at the declared dependency floors and at current releases.
