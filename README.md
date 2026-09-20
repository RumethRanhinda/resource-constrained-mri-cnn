# EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification

This repository contains the source code, data processing scripts, and evaluation notebooks for Assignment 03. Our goal is to build, train, and evaluate a lightweight Convolutional Neural Network (CNN) optimized for low-resource embedded devices. 

## Project Objectives

1.  **Data Preparation:** Emulate a low-resource sensor environment by downscaling our chosen dataset to a maximum resolution of $64\times64$ pixels, split into 70% training, 15% validation, and 15% testing sets.
2.  **Model A (Standard CNN):** A baseline model interleaving standard 2D convolutional and max-pooling layers.
3.  **Model B (Lightweight CNN):** A custom architecture utilizing Depthwise Separable Convolutions, strictly constrained to under 100,000 trainable parameters.
4.  **SOTA Comparison:** Fine-tuning two state-of-the-art lightweight edge models (e.g., MobileNetV2, SqueezeNet) to compare accuracy, memory footprint, and computational cost against our custom Model B.

## Repository Structure

```text
EN3150_CNN_Image_Classification/
├── data/                  # Ignored by Git. Store 64x64 dataset splits here.
├── notebooks/             # Jupyter notebooks for data exploration and optimizer tuning.
├── src/
│   ├── data_prep.py       # Scripts for downloading, resizing (64x64), and splitting data.
│   ├── model_a.py         # Standard CNN architecture.
│   ├── model_b.py         # Sub-100k parameter lightweight CNN.
│   └── train.py           # Training loops, validation, and evaluation metrics.
├── .gitignore             # Prevents committing large data files and __pycache__.
├── requirements.txt       # Python dependencies.
├── LICENSE.md             # Project MIT license
└── README.md              # Project documentation.
```

## Local Environment Setup

To ensure everyone is working with the same dependency versions and to prevent environment conflicts, please set up a localized Python virtual environment before running any code.

### 1. Clone the Repository
```bash
git clone https://github.com/RumethRanhinda/CNN-Img-classification
cd EN3150_CNN_Image_Classification
```

### 2. Create the Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate the Environment
*   **For Ubuntu/Linux:**
    ```bash
    source .venv/bin/activate
    ```
*   **For Windows:**
    ```bash
    .venv\Scripts\activate
    ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Workflow & Git Protocols

*   **Commit Regularly:** The grading rubric explicitly evaluates our GitHub commit history to ensure steady progress. 
*   **Data Handling:** Do not commit raw or processed image datasets to the repository. The `.gitignore` is configured to ignore the `data/` directory. If you write a new data loading script, commit the script, not the data.
*   **Branching:** If you are testing a new architecture or optimizer, create a separate branch and merge it into `main` once it is stable.
