# EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification

This repository contains the end-to-end implementation for **EN3150 Assignment 03**. Our objective is to design, train, and evaluate lightweight Convolutional Neural Networks (CNNs) optimized for resource-constrained edge computing devices (microcontrollers, Raspberry Pi, and edge sensor nodes).

---

## Table of Contents
1. [Project Overview & Assignment Constraints](#project-overview--assignment-constraints)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Architectures](#model-architectures)
6. [Training & Evaluation](#training--evaluation)
7. [Upcoming Tasks for Teammates](#upcoming-tasks-for-teammates)
8. [Git Protocols & Workflow](#git-protocols--workflow)

---

## Project Overview & Assignment Constraints

The goal of this assignment is to explore the trade-offs between **model capacity**, **computational complexity (MACs/FLOPs)**, and **real-world inference latency** on edge hardware:
- **Low-Resolution Emulation:** Images downscaled to a maximum of $64 \times 64$ pixels to emulate low-resource sensor feeds.
- **Dataset Partitioning:** Strict split into **70% training**, **15% validation**, and **15% testing**.
- **Model A (Baseline):** Standard interleaved 2D convolutions and max-pooling layers.
- **Model B (Lightweight):** Depthwise Separable Convolutions to drastically reduce Multiply-Accumulate (MAC) operations.
  - **Hard Constraint:** Total trainable parameters **must not exceed 100,000**.
- **Hardware-Aware Design:** Rationale for activation functions (e.g., using single-instruction `ReLU` instead of computationally expensive exponentials like `sigmoid`/`tanh`).

---

## Repository Structure

```text
CNN-Img-classification/
├── data/
│   ├── OriginalDataset/            # Raw MRI dataset folders (Ignored by Git)
│   │   ├── NonDemented/
│   │   ├── VeryMildDemented/
│   │   ├── MildDemented/
│   │   └── ModerateDemented/
│   └── processed/                  # Processed 64x64 splits (Ignored by Git)
│       ├── train/                  # 70% split (2,240 Demented, 2,240 NonDemented)
│       ├── val/                    # 15% split (480 Demented, 480 NonDemented)
│       └── test/                   # 15% split (480 Demented, 480 NonDemented)
├── docs/
│   └── EN3150_Assignment_03_In23.pdf # Assignment specification sheet
├── models/                         # Saved trained model files (.keras) (Ignored by Git)
├── notebooks/                      # Exploratory data analysis & experimental notebooks
├── reports/
│   └── figures/                    # Generated learning curves and confusion matrices
├── src/
│   ├── prepare_dataset.py          # Resizes raw images to 64x64 & creates 70/15/15 splits
│   ├── model_a.py                  # Model A: Standard Baseline CNN (~228k parameters)
│   ├── model_b.py                  # Model B: Lightweight Edge CNN (~63k parameters, <100k)
│   └── train.py                    # Training pipeline, validation, evaluation & metrics
├── .gitignore                      # Prevents committing large data and model weight files
├── requirements.txt                # Required Python packages
└── README.md                       # Workspace setup & execution guide
```

---

## Environment Setup

To ensure consistent dependencies across all group members, please use a localized Python virtual environment (`.venv`).

### 1. Clone the Repository
```bash
git clone https://github.com/RumethRanhinda/CNN-Img-classification.git
cd CNN-Img-classification
```

### 2. Create the Virtual Environment
Make sure you have Python 3.10+ installed:
```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment
- **Windows (Command Prompt / PowerShell):**
  ```powershell
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

We use an Alzheimer's MRI dataset mapped into a balanced binary classification task:
- **`NonDemented`** $\to$ `NonDemented`
- **`VeryMildDemented`**, **`MildDemented`**, **`ModerateDemented`** $\to$ `Demented`

### Step 1: Place Raw Images
Ensure your raw uncompressed images are placed in:
```text
data/OriginalDataset/
    ├── NonDemented/
    ├── VeryMildDemented/
    ├── MildDemented/
    └── ModerateDemented/
```

### Step 2: Run Preprocessing Script
Run the automated preprocessing script:
```bash
python src/prepare_dataset.py
```

**What this script does:**
1. Resizes all images to **$64 \times 64$ pixels** using high-quality Lanczos resampling.
2. Applies a fixed random seed (`seed=42`) for 100% reproducible splits across teammates.
3. Automatically creates `data/processed/train/` (70%), `data/processed/val/` (15%), and `data/processed/test/` (15%).

---

## Model Architectures

### Model A: Standard Baseline CNN ([src/model_a.py](src/model_a.py))
- Interleaves four standard `Conv2D` blocks ($16 \to 32 \to 64 \to 128$) with `MaxPooling2D`.
- Classification head: `Flatten` $\to$ `Dense(64)` $\to$ `Dropout(0.3)` $\to$ `Dense(2)`.
- **Total Parameters:** $228,418$ ($\approx 892\text{ KB}$ weights).

### Model B: Lightweight Edge CNN ([src/model_b.py](src/model_b.py))
- Uses **Depthwise Separable Convolutions** (`SeparableConv2D`) to decouple spatial filtering from cross-channel mixing, yielding an $85\text{--}90\%$ reduction in MAC operations.
- Interleaves four separable blocks ($32 \to 64 \to 128 \to 256$) with `BatchNormalization` and `MaxPooling2D`.
- Classification head: `GlobalAveragePooling2D` $\to$ `Dense(64)` $\to$ `Dropout(0.3)` $\to$ `Dense(2)`.
- **Total Parameters:** $64,290$ (Trainable: **$63,330$**, strictly complying with the $<100,000$ constraint).

---

## Training & Evaluation

The script [src/train.py](src/train.py) provides an automated pipeline for training, validation monitoring, model saving, and test evaluation.

### Basic Commands

- **Train both Model A and Model B:**
  ```bash
  python src/train.py --model both --epochs 30
  ```

- **Train only Model A:**
  ```bash
  python src/train.py --model a --epochs 30
  ```

- **Train only Model B:**
  ```bash
  python src/train.py --model b --epochs 30
  ```

### CLI Arguments & Options
- `--epochs`: Maximum number of training epochs (default: `20`).
- `--batch_size`: Batch size for data loaders (default: `32`).
- `--no-early-stopping`: Disables `EarlyStopping` and `ReduceLROnPlateau` (trains for exact fixed epoch count).

### Outputs Generated Automatically
1. **Saved Models:** Stored in `models/` as `model_a.keras` and `model_b.keras`.
2. **Learning Curves:** Train/Val Loss and Accuracy plots saved to `reports/figures/model_a_learning_curves.png` and `reports/figures/model_b_learning_curves.png`.
3. **Confusion Matrices:** Evaluated on the test split and saved to `reports/figures/*_confusion_matrix.png`.
4. **Console Summary:** Displays accuracy, precision, recall, confusion matrix, and a side-by-side comparison table.

---

## Upcoming Tasks for Teammates

Here are the remaining tasks from the assignment sheet to distribute:

1. **Section 3: Optimizer Tuning & Comparison Study [15 Marks]**
   - Compare our chosen optimizer (**Adam**, lr=0.001) against:
     - (a) Standard **SGD** (momentum = 0.0)
     - (b) **SGD with Momentum** (momentum = 0.9)
   - Plot loss trajectories and document the theoretical impact of the momentum parameter on overcoming local saddle points and accelerating convergence.

2. **Section 5: SOTA Lightweight Network Fine-Tuning [20 Marks]**
   - Select two pre-trained edge models (e.g., **MobileNetV2** and **EfficientNet-B0**).
   - Adapt input shape for $64 \times 64$ and fine-tune using the exact same train/val/test splits.
   - Record parameter counts, model size (MB), test accuracy, precision, and recall.

3. **Section 6: Final Comparison & Report Writing [20 Marks]**
   - Synthesize comparison tables across Model A, Model B, and the two SOTA models.
   - Analyze trade-offs across the triangle: **Accuracy vs. Memory Footprint vs. Computational Cost**.
   - Compile the final PDF report (`<Yourgroupno>_A03_EN3150.pdf`).

---

## Git Protocols & Workflow

- **Regular Commits:** The assignment grading explicitly evaluates commit frequency over time. Make small, focused commits as you work on your parts.
- **Do NOT Commit Data or Models:** The `.gitignore` is already set up to ignore `data/` and `models/`. Never force-add raw datasets or binary weight checkpoints.
- **Branching:** Work on separate feature branches (e.g., `feature/optimizer-study`, `feature/sota-models`) and open pull requests before merging into `main`.
