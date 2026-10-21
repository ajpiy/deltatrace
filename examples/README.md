# Examples

| Notebook | Modality | Description |
| :--- | :--- | :--- |
| [`quickstart.ipynb`](quickstart.ipynb) | Synthetic benchmark | **Start here.** Self-contained walkthrough: spatial graph blending (`p_keep`), trajectory inference, streamline velocity fields, and unsupervised sink discovery on synthetic crypt data. Runs in seconds with no external files. |
| [`spatial_transcriptomics.ipynb`](spatial_transcriptomics.ipynb) | Spatial transcriptomics | Template for single-cell-resolution ST (MERFISH, Stereo-seq, Xenium, Slide-seq). Demonstrates pairwise spatial distance calculation, endpoint sampling, QROT flow, and streamline velocity fields. Pre-rendered on axolotl brain Stereo-seq data. |
| [`single_cell_rna_seq.ipynb`](single_cell_rna_seq.ipynb) | scRNA-seq + paired ST | Template for dissociated scRNA-seq without tissue coordinates. Demonstrates cell-to-spot mapping (e.g., Tangram), Wasserstein distance (`dt.pp.wasserstein`), and cluster trajectory abstraction. Pre-rendered on mouse liver differentiation. |

## Data

- **`quickstart.ipynb`**: Self-contained; generates its own data via `dt.datasets.crypt()`.
- **`spatial_transcriptomics.ipynb`**: Template pre-rendered on axolotl Stereo-seq data; point `data_path` at your AnnData with coordinates in `obsm["spatial"]`.
- **`single_cell_rna_seq.ipynb`**: Template pre-rendered on mouse liver; point at your scRNA-seq and spatial AnnData files (Tangram is optional; install via `pip install tangram-sc` if cell-to-spot mapping is needed).
