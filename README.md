<p align="center">
  <img src="assets/DeltaTrace_cover.png" alt="DeltaTrace" width="520">
</p>

<p align="center">
  <em>spatially regularized trajectory inference for single-cell and spatial transcriptomics</em>
</p>

## Overview
![Visual Abstract](assets/method_overview.png)

**DeltaTrace** is a trajectory inference (TI) method regularized by spatial distance for single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics (ST) data. It infers the trajectory of cells and their respective lineage assignment using Essid and Solomon's quadratically regularized graph optimal transport ([QROT](https://doi.org/10.48550/arXiv.1704.08200)).

**DeltaTrace** is designed to run on an **integrated** cell graph, where two cells stay connected only if they are close both in expression and in physical space. The same pipeline also runs on the expression kNN graph alone, e.g. when no spatial prior is available.

## Key Innovations

1. scRNA-seq cells carry no tissue coordinates. Given a cell-to-spot mapping onto a paired ST section (from any probabilistic mapping method), **DeltaTrace** takes the *Earth Mover's distance* between cells' resulting spatial distributions as an inferred physical proximity.

2. For TI, transport runs on the integrated cell graph rather than a learned embedding. QROT returns a dual potential (taken directly as pseudotime) together with the primal directed flow, solved through its dual problem: extremely fast with little memory useage, requiring no time-series data or neural network to train.

## Installation

**DeltaTrace** requires Python >= 3.12.

<!-- Once on PyPI:
```bash
pip install deltatrace
```
-->

```bash
git clone https://github.com/ajpiy/deltatrace.git
cd deltatrace
pip install .
```

Use `uv sync` instead for development checkout.

## Inputs

- **`adata`**: preprocessed cells (`AnnData`), with cluster labels in `obs["clusters"]` (or pass `cluster_key=`) and a scanpy neighbor graph (from `scanpy.pp.neighbors`).
- **`adata_dist`** (optional): the cells' inferred or observed secondary distances, as an `AnnData` whose `X` holds the `(n_cells, n_cells)` matrix (or a labeled `DataFrame`, or the condensed output of `dt.pp.wasserstein`), indexed by `adata`'s cell names (matched automatically):
  - *ST at cell resolution*: the Euclidean distance between cell coordinates ([`examples/spatial_transcriptomics.ipynb`](examples/spatial_transcriptomics.ipynb)).
  - *scRNA-seq with a paired ST section*: map cells to spots (any probabilistic mapper, e.g. Tangram), then `dt.pp.wasserstein` ([`examples/single_cell_rna_seq.ipynb`](examples/single_cell_rna_seq.ipynb)).

`dt.datasets.crypt()` returns a small synthetic pair in this form: a crypt-shaped tissue whose two walls run the same expression program, with `["progenitor"]` as the source and `["fate_A", "fate_B"]` as the sinks. [`examples/quickstart.ipynb`](examples/quickstart.ipynb) walks through it.

## Usage

```python
import deltatrace as dt

# builds the expression and spatially integrated graphs into one object
ad_intg = dt.pp.integrate(adata, adata_dist)

# runs: endpoints -> qrot -> velocity -> lineages on the intersection graph (intr)
# sources / sinks: lists of progenitor and terminal cluster labels
dt.tl.trajectory(ad_intg, sources, sinks, graph="intr")

# pseudotime with the flow field
dt.pl.pseudotime(ad_intg)
```
<!---
No annotated sinks? Pass `None` as the sinks to run source-only: every non-source cell absorbs the flow, and `trajectory` ranks the clusters by terminality so you can pick the fates.

### QROT on any graph

The solver also runs outside the cell pipeline, on any graph in SciPy's csgraph form:

```python
# graph: (n, n) matrix of edge costs, graph[i, j] the cost of edge i -> j
# a, b: (n,) source and sink masses, equal totals on each connected piece
res = dt.tl.qrot_graph(graph, a, b)    # edges run both ways unless directed=True

res["pseudotime"]    # (n,) in [0, 1], NaN where a node has no potential
res["flow"]          # (n, n) sparse array of the optimal flow
```

`qrot_graph`'s default `tau=100` is set for masses that total 1, and costs are divided by the largest first; `tau="auto"` (the default of `tl.qrot`) takes 0.3 times the nodes in the largest connected piece.
-->

## Public API

**DeltaTrace** uses [scanpy](https://doi.org/10.1186/s13059-017-1382-0)'s namespace layout, and every stage `tl.trajectory` composes is also callable on its own:

| Namespace | Functions |
|-----------|-----------|
| `dt.pp` (preprocess) | `wasserstein`, `integrate` |
| `dt.tl` (tool) | `trajectory`, `endpoints`, `qrot`, `qrot_graph`, `compute_velocity_grid`, `lineages`, `terminality`, `widest_path`, `cluster_trajectory` |
| `dt.pl` (plot) | `pseudotime`, `velocity`, `endpoints`, `lineages`, `cluster_trajectory` |
| `dt.datasets` | `crypt` |

## Citation

Manuscript is in preparation. Until its release, please cite the software via
[`CITATION.cff`](CITATION.cff) (GitHub renders it at "Cite this repository" under the "About" section).

## License

See [LICENSE](LICENSE).
