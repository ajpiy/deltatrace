# Examples

| Notebook | What it shows |
|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | **Start here.** The pipeline on the synthetic `dt.datasets.crypt()`, what the spatial constraint changes, and finding fates without sinks. Needs nothing but the package. |
| [`spatial_transcriptomics.ipynb`](spatial_transcriptomics.ipynb) | Single-cell-resolution spatial transcriptomics, from cell coordinates to trajectory, running the stages by hand, with source-only discovery. |
| [`single_cell_rna_seq.ipynb`](single_cell_rna_seq.ipynb) | Template for scRNA-seq with a paired spatial section; cell-to-spot mapping, then `pp.wasserstein`. |

## Data

The quickstart makes its own data. The spatial dataset will be released with the manuscript (in
preparation). The scRNA-seq notebook is a template; point it at your data (Tangram is not a
dependency; `pip install tangram-sc`).
