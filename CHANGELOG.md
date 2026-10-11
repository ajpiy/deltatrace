# Changelog

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
