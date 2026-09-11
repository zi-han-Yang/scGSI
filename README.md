# scGSI: Graph-guided self-supervised integration of paired single-cell multi-omics

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**scGSI** is a graph-guided self-supervised deep learning framework tailored for the efficient integration of paired single-cell multi-omics data. The framework constructs heterogeneous multi-graph representations to encapsulate modality-specific topological architectures, employs a pull-in projection mechanism to pre-optimize inter-modality affinities, and utilizes a specialized cross-fusion module to orchestrate feature interplay.

This repository contains the implementation of scGSI along with comprehensive evaluation scripts.

## Framework

The overall architecture of scGSI is illustrated in the framework diagram below:

<div align="center">
  <img src="fig-framework.png" alt="scGSI Framework" width="800"/>
</div>


## Authors

- **Xiang Chen**
- **Zihan Yang**
- **Xiaoyu Liu**
- **Zhiyi Xie**
- **Wenlu Guo**

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.8 or higher
- CUDA (optional, for GPU acceleration)

### Dependencies

Install the required packages using the requirements file:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torch-geometric numpy scipy pandas scanpy anndata scikit-learn matplotlib seaborn tqdm pyyaml munkres h5py
```

**Note:** For GPU support, install PyTorch with CUDA from the [official PyTorch website](https://pytorch.org/).

### Installation from Source

You can also install scGSI as a package:

```bash
pip install -e .
```

## Project Structure

```
scGSI/
├── scGSI/              # Main model implementation
│   ├── model.py         # scGSI model architecture
│   ├── layers.py        # Neural network layers (GraphSAGE, GATv2, etc.)
│   ├── losses.py        # Loss functions
│   ├── dataset.py       # Data loading utilities
│   ├── preprocess.py    # Data preprocessing
│   ├── evaluate.py      # Evaluation metrics
│   ├── utils.py         # Utility functions
│   └── main.py          # Main training script
├── config/              # Configuration files for different datasets
│   └── Tea_PBMC.yaml
├── data/                # Data loading scripts
├── eva/                 # Evaluation scripts
├── vis/                 # Visualization scripts
├── logs/                # Model checkpoints (gitignored)
├── .gitignore           # Git ignore rules
├── .gitattributes       # Git attributes for line endings
├── requirements.txt     # Python dependencies
├── setup.py             # Package setup script
├── LICENSE              # MIT License
├── CONTRIBUTING.md      # Contribution guidelines
├── fig-framework.png    # Framework diagram
└── README.md            # This file
```

## Usage

### Data Preparation

Prepare your data in AnnData (`.h5ad`) format:
- RNA-seq data: `raw_data_rna.h5ad`
- ATAC-seq data: `raw_data_atac.h5ad`

Each AnnData object should contain:
- `.X`: Feature matrix (cells × features)
- `.obs['clusters']`: Cell clusters labels

### Configuration

Edit the configuration file in `config/` directory for your dataset. Example configuration (`config/Tea_PBMC.yaml`):

```yaml
dataset_name: Tea_PBMC
dataset_dir: ../data/Tea_PBMC/
dataset_type: RNA_ATAC
GAM_name: Signac
paired: True

n_high_var: 2000
hid_dim: 256
neighbors_mnn: 200
metric: minkowski
use_rep: hvg_count

latent_dim: 32
k_neighbors: 10
k_clusters: 11

seed: 666
batch_size: 256
learning_rate: 0.003
weight_decay: 0.00001
epoch: 300
```

### Training

Run the main training script:

```bash
cd scGSI
python main.py
```

Or modify the dataset name in `main.py`:

```python
dataset_name = "Tea_PBMC"  # or "CITE_PBMC", "AdBraCor", "P0BraCor", "BMMC_s1d1"
```

### Evaluation

Run evaluation scripts:

```bash
cd eva
python eva_Tea_PBMC.py  # Replace with your dataset name
```

### Visualization

Generate visualizations:

```bash
cd vis
python visual_Tea_PBMC.py  # Replace with your dataset name
```

## Citation

If you use **scGSI** in your research, including its source code, model, methodology, data-processing pipeline, evaluation scripts, visualization scripts, or other materials associated with this project, please cite our paper.

**Citation:** Chen X, Yang Z, Liu X, Xie Z, Guo W (2026) scGSI: Graph-guided self-supervised integration of paired single-cell multi-omics. *PLoS Computational Biology* 22(9): e1014773. https://doi.org/10.1371/journal.pcbi.1014773

### BibTeX

```bibtex
@article{chen2026scgsi,
  title   = {scGSI: Graph-guided self-supervised integration of paired single-cell multi-omics},
  author  = {Chen, Xiang and Yang, Zihan and Liu, Xiaoyu and Xie, Zhiyi and Guo, Wenlu},
  journal = {PLOS Computational Biology},
  volume  = {22},
  number  = {9},
  pages   = {e1014773},
  year    = {2026},
  doi     = {10.1371/journal.pcbi.1014773}
}
```

For publications, presentations, benchmarks, or other academic work based on scGSI, we kindly ask you to include the citation above to acknowledge the original work.

**Copyright:** © 2026 Chen et al. This is an open access article distributed under the terms of the [Creative Commons Attribution License](http://creativecommons.org/licenses/by/4.0/), which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## Acknowledgments

We thank the developers of the comparison methods and the single-cell genomics community for their valuable contributions.

