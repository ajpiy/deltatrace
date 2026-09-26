# Changelog

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
