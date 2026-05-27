# Intelligent Cigar Wrapper Grading Based on an Improved ResNet Framework

[License: MIT](https://opensource.org/licenses/MIT)
[PyTorch](https://pytorch.org/)

Official PyTorch implementation for the paper: **"[Adaptive Classification and Grading Model of Cigar wrapper leaf Based on Improved ResNet Algorithm]"**. 

## 📝 Overview

Automated grading of cigar wrapper leaves is a challenging fine-grained visual classification task due to subtle local defects, class imbalance, and complex background interference. This repository provides a robust, industrial-grade framework designed to tackle these challenges. 

Our proposed method integrates:

- **Mask4Ch:** A 4th-channel binary mask to explicitly guide foreground spatial attention.
- **WRS (Two-Stage Weighted Random Sampling):** Mitigates class imbalance during early training.
- **EMA (Exponential Moving Average):** Stabilizes network weights.
- **Dual-Head Loss (CE + CORAL):** Captures both categorical boundaries and ordinal relationships.

Through extensive comparative experiments across distinct architectural paradigms, we demonstrate that orthodox dense CNNs (e.g., **ResNet-50**) natively synergize with our pixel-level Mask4Ch module, achieving state-of-the-art accuracy, whereas modern "Patchify" architectures (e.g., ViT, Swin-T, ConvNeXt) mechanically fracture the spatial priors.

## 📂 Repository Structure

```text
Cigar-Wrapper-Grading/
├── data/                            # Sample images and dataset guidelines
├── scripts/                         # Bash scripts for automated experiments
│   ├── run_ablation.sh              # Ablation studies script
│   └── run_cross_model.sh           # Cross-architecture validation script
├── main.py                          # Unified training and evaluation script
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation

```

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/Ison-20/Cigar_classification.git.

```

1. Install the required dependencies:

```bash
pip install -r requirements.txt

```

*(Main dependencies include `torch`, `torchvision`, `timm`, `opencv-python`, and `scikit-learn`.)*

## 🗂️ Data Preparation

Due to industrial confidentiality agreements, the full FX-01 cultivar dataset (~8.6k images) is not publicly available. We provide a few desensitized sample images in `data/sample_images/` for code testing.

To train on your own custom dataset, please organize your images in the standard PyTorch `ImageFolder` format:

```text
data/JYB
├── JYB1/
├── JYB2/
└── JYB3/

```

## 🚀 Quick Start

### 1. Standard Training (Our Proposed Framework)

Run `main.py` with the optimal hyperparameters (A4 configuration) reported in the paper:

```bash
python main.py \
    --data ./data/sample_images \
    --model_name resnet50 \
    --epochs 50 \
    --batchsize 32 \
    --lr 1e-4 \
    --add_mask_channel \
    --use_weighted_sampler --switch_off_sampler_epoch 11 \
    --use_ema --ema_decay 0.995 --ema_start_epoch 3 \
    --use_ce --use_coral --ce_weight 0.7 --coral_weight 0.3 \
    --max_rotate 10

```

### 2. Reproducing Cross-Model Comparisons

To reproduce the architectural comparison table from our paper, run the provided bash script:

```bash
bash scripts/run_cross_model.sh ./data/custom_dataset

```

## 📊 Main Results

Our multi-module framework significantly improves the performance of orthodox CNNs while revealing the architectural incompatibility of "Patchify" stems in modern vision models.


| Backbone Paradigm                          | Model          | Config / Method       | Accuracy | Macro-F1 | QWK     |
| ------------------------------------------ | -------------- | --------------------- | -------- | -------- | ------- |
| **Orthodox CNNs**<br>*(Preserves Mask boundary)* | ResNet-50      | Baseline              | 0.9065   | 0.9157   | 0.9543  |
|                                            | ResNet-50      | **Ours (Framework)**  | **0.9439** | **0.9500** | **0.9638** |
|                                            | DenseNet-121   | Baseline              | 0.9221   | 0.9332   | 0.9487  |
|                                            | DenseNet-121   | **Ours (Framework)**  | **0.9439** | **0.9518** | **0.9500** |

## 🎓 Citation

If you find this code or our conceptual insights helpful for your research, please consider citing our paper:

```bibtex
@article{Du2026Cigar,
  title={Adaptive Classification and Grading Model of Cigar Wrapper Leaf Based on Improved ResNet Algorithm},
  author={Du, Chaofan and Wang, Ruiqi and Wu, Tianyi},
  journal={Scientific Reports},
  year={2026},
  publisher={Nature Portfolio}
}

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
