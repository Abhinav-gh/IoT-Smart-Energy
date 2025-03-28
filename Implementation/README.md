# ASHRAE Great Energy Predictor III - Implementation Guide

Welcome to the **ASHRAE Prediction Pipeline**! This directory contains all the necessary scripts to preprocess data, train models using LightGBM with K-fold validation, and generate predictions.

---

## 📖 Table of Contents
- [ASHRAE Great Energy Predictor III - Implementation Guide](#ashrae-great-energy-predictor-iii---implementation-guide)
  - [📖 Table of Contents](#-table-of-contents)
  - [📂 Directory Structure](#-directory-structure)
  - [🚀 How to Use the Pipeline](#-how-to-use-the-pipeline)
    - [**1️⃣ Setup Environment**](#1️⃣-setup-environment)
    - [**2️⃣ Running the Pipeline**](#2️⃣-running-the-pipeline)
      - [**Optional Arguments:**](#optional-arguments)
    - [**3️⃣ Understanding the Pipeline Steps**](#3️⃣-understanding-the-pipeline-steps)
      - [🔹 Step 0: Dataset Preparation](#-step-0-dataset-preparation)
      - [🔹 Step 1: Preprocessing](#-step-1-preprocessing)
      - [🔹 Step 2: Model Training (LightGBM with K-Fold)](#-step-2-model-training-lightgbm-with-k-fold)
      - [🔹 Step 3: Inference](#-step-3-inference)
  - [� Notes](#-notes)
  - [❓ Need Help?](#-need-help)
  - [👤 Author](#-author)

---

## 📂 Directory Structure

```
📁 Implementation
│── 📁 Pre-Processing
│   ├── preprocessing.py   # Preprocessing script (supports faster mode)
│── 📁 K_fold_LGBM
│   ├── train.py           # LightGBM model training
│   ├── inference.py       # Inference script to generate predictions
│── prepare_data.py        # Dataset preparation
│── run_pipeline.sh        # Main script to run the entire pipeline
│── README.md              # This documentation
```

---

## 🚀 How to Use the Pipeline

### **1️⃣ Setup Environment**
Ensure you have the required dependencies installed. If you use **conda**, create an environment:

```bash
conda env create -f environment.yml
conda activate Ashrae_Predictor
```

If using **pip**, install dependencies manually:
```bash
pip install -r requirements.txt
```

---

### **2️⃣ Running the Pipeline**
The entire pipeline can be executed using the `run_pipeline.sh` script.

```bash
bash run_pipeline.sh
```

#### **Optional Arguments:**
| Argument  | Description |
|-----------|------------|
| `--faster` | Uses reduced dataset for model. (around 25% of original size) |
| `--help`   | Displays usage instructions. |

Example:
```bash
bash run_pipeline.sh --faster
```

---

### **3️⃣ Understanding the Pipeline Steps**

#### 🔹 Step 0: Dataset Preparation
```bash
python prepare_data.py
```
- Prepares the dataset before processing begins.

#### 🔹 Step 1: Preprocessing
```bash
python Pre-Processing/preprocessing.py
```
- Cleans and processes raw data.
- Supports a `--faster` mode for reduced dataset preprocessing.

#### 🔹 Step 2: Model Training (LightGBM with K-Fold)
```bash
python K_fold_LGBM/train.py
```
- Trains LightGBM using K-fold cross-validation.
- Saves trained models for inference.

#### 🔹 Step 3: Inference
```bash
python K_fold_LGBM/inference.py
```
- Generates predictions using the trained models.
- Outputs final results.

---

<!-- ## 📊 Model Performance
| Model | K-Fold Score (CV) | Test Score (RMSE) |
|--------|------------------|------------------|
| LightGBM | **X.XXX** | **X.XXX** |
| Baseline | **X.XXX** | **X.XXX** |

> *(Replace with actual results after training)* -->

---

## 📌 Notes
- Make sure all dependencies are installed before running.
- The preprocessing step must be completed before training.
- If using a different Python interpreter, the script will prompt for the correct one.

---

## ❓ Need Help?
Run the following command to view available options:
```bash
bash run_pipeline.sh --help
```

For any issues, refer to the documentation or raise a question in the project repository.

---

## 👤 Author  
**Abhinav Deshpande** [@Abhinav-gh](https://github.com/Abhinav-gh)
