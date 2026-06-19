# SRRMRec

Official PyTorch implementation of **Structural Reliability Reweighting for Multimodal Recommendation**.

SRRMRec improves multimodal recommendation by refining noisy item-item graphs. It estimates the reliability of each graph edge from local topology, softly suppresses unreliable connections, and strengthens trustworthy relations. A cross-view self-supervised objective further aligns collaborative and multimodal item representations.

## Overview

SRRMRec contains three main components:

- **Multimodal graph construction:** builds visual and textual KNN graphs from pretrained item features.
- **Structural Reliability Reweighting (SRR):** adjusts edge weights using topology-aware metrics, including Adamic-Adar, Resource Allocation, Jaccard, and Local Path.
- **Cross-view self-supervision:** aligns collaborative filtering and multimodal representations with an InfoNCE loss.

The model is jointly optimized with BPR loss, regularization, and self-supervised loss.

## Requirements

The original environment uses:

```text
Python 3.7.11
PyTorch 1.11.0
NumPy 1.21.5
Pandas 1.3.5
SciPy 1.7.3
PyYAML 6.0
```

Install the core dependencies with:

```bash
conda create -n srrmrec python=3.7.11
conda activate srrmrec
pip install torch==1.11.0 numpy==1.21.5 pandas==1.3.5 scipy==1.7.3 pyyaml==6.0
pip install matplotlib pillow lmdb torchvision
```

Install the PyTorch build that matches your CUDA version if needed. The `requirements.txt` file records the original package versions, including the Python version used by the authors.

## Datasets

The paper evaluates SRRMRec on three Amazon datasets:

| Dataset | Users | Items | Interactions |
| --- | ---: | ---: | ---: |
| Baby | 19,445 | 7,050 | 160,792 |
| Sports | 35,598 | 18,357 | 296,337 |
| Clothing | 39,387 | 23,033 | 278,677 |

Preprocessed interactions and multimodal features can be downloaded from the [MMRec dataset repository](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing).

Place the downloaded files under `data/`:

```text
data/
├── baby/
│   ├── baby.inter
│   ├── image_feat.npy
│   └── text_feat.npy
├── sports/
└── clothing/
```

The interaction file must contain `userID`, `itemID`, and `x_label`, where labels `0`, `1`, and `2` represent the training, validation, and test sets.

## Usage

Run the program from the `src` directory:

```bash
cd src
python main.py --model SRRMRec --dataset baby
```

Available paper datasets are `baby`, `sports`, and `clothing`.

The main configuration files are:

```text
src/configs/overall.yaml
src/configs/model/SRRMRec.yaml
src/configs/dataset/
```

To run on CPU, set the following option in `overall.yaml`:

```yaml
use_gpu: False
```

## Main Hyperparameters

| Parameter | Description |
| --- | --- |
| `knn_k` | Number of neighbors in the multimodal KNN graph |
| `beta` | Strength of structural reliability reweighting |
| `reliability_metric` | Reliability metric: `AA`, `RA`, `Jaccard`, or `LP` |
| `n_mm_layers` | Number of multimodal graph propagation layers |
| `n_ui_layers` | Number of user-item graph propagation layers |
| `ssl_temp` | Temperature used by the InfoNCE loss |
| `ssl_weight` | Weight of the self-supervised objective |

The paper uses 64-dimensional embeddings, a batch size of 2048, and early stopping based on Recall@20.

## Results

SRRMRec achieves competitive or state-of-the-art performance on Baby, Sports, and Clothing. The experiments show that:

- structural reweighting consistently improves the original multimodal graph;
- all four reliability metrics are effective;
- self-supervised alignment provides a substantial performance gain;
- the method introduces limited memory and training overhead.

## Project Structure

```text
├── data/                       # Datasets and multimodal features
├── src/
│   ├── common/                 # Training framework
│   ├── configs/                # Model and dataset configurations
│   ├── models/srrmrec.py       # SRRMRec implementation
│   ├── utils/                  # Data loading and evaluation
│   └── main.py                 # Entry point
├── requirements.txt
└── README.md
```

## Acknowledgements

This project is implemented based on the [MMRec](https://github.com/enoche/MMRec) framework.

## Citation

If this work is useful to your research, please cite:

```bibtex
@article{chen_srrmrec,
  title   = {Structural Reliability Reweighting for Multimodal Recommendation},
  author  = {Chen, Jicong and Zhang, Yihao and He, Qinyang and Liu, Jiangchuan and Zhou, Wei},
  note    = {Preprint}
}
```
