# False Data Injection Detection in Smart Grids

> A deep learning-based system for detecting False Data Injection (FDI) attacks in smart grid sensor networks, featuring conformal prediction scoring and a real-time Gradio web interface.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Dataset Generation](#dataset-generation)
  - [Model Architecture](#model-architecture)
  - [Conformal Prediction Scoring](#conformal-prediction-scoring)
  - [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Generate Dataset](#step-1-generate-dataset)
  - [Step 2: Train & Evaluate](#step-2-train--evaluate)
  - [Step 3: Live Inference via Gradio](#step-3-live-inference-via-gradio)
- [File Descriptions](#file-descriptions)
- [Model Performance](#model-performance)
- [Sample Inputs](#sample-inputs)
- [Output Files](#output-files)
- [Logging](#logging)

---

## Overview

False Data Injection (FDI) attacks are a critical threat to smart grid infrastructure. Adversaries tamper with sensor measurements — such as power flow, voltage, and temperature readings — to deceive control systems. This project implements a **binary classifier** using a deep neural network (DNN) to distinguish between **legitimate (legit)** sensor readings and **injected false data**.

Key capabilities:
- Generates synthetic smart grid datasets with configurable false-data injection ratios
- Trains a 4-layer DNN with Batch Normalization, Swish/LeakyReLU activations, and Dropout
- Evaluates the model using Accuracy, Precision, Recall, F1 Score, AUC-ROC, and Sensivity
- Computes **Conformity Scores** per sample for uncertainty quantification
- Launches a **Gradio web UI** for real-time live inference
- Logs all dataset loading and inference events to `model_logs.log`

---

## Project Structure

```
finall/
│
├── r3_gradio.py            # Main script: training pipeline + Gradio UI
├── dataset.py              # Synthetic dataset generator (Gaussian noise-based)
├── standardization.py      # Real-data preprocessor with false data injection
├── normal_to_stan.py       # Utility: manual z-score standardization example
│
├── data.csv                # Raw real-world sensor data (input for standardization.py)
├── train.csv               # Generated training split
├── test.csv                # Generated test split
│
├── values_true.txt         # Example standardized legit sensor readings
├── values_false.txt        # Example injected/anomalous sensor readings
│
├── plots/
│   ├── training_history.png        # Accuracy & loss curves over epochs
│   ├── conformity_distribution.png # Histogram of conformity scores
│   └── roc_curve.png               # ROC curve with AUC annotation
│
├── model_report_<timestamp>.csv    # Per-run evaluation metrics
├── conformity_scores_<timestamp>.csv # Per-sample conformity scores
└── model_logs.log                  # Runtime logs
```

---

## How It Works

### Dataset Generation

Two approaches are provided:

**Synthetic (dataset.py)**
- Generates 5,000 samples with 5 features sampled from N(0, 1)
- Injects false data into 15% of samples by adding Gaussian noise N(0, 2) to existing legit samples
- Splits data 70/30 into `train.csv` and `test.csv`

**Real Data (standardization.py)**
- Loads real sensor measurements from `data.csv`
- Standardizes features using `StandardScaler` (zero mean, unit variance)
- Injects false data at a 15% ratio using the same noise perturbation method
- Saves as `train_standardized.csv` and `test_standardized.csv`

### Model Architecture

The DNN is a sequential binary classifier with the following layers:

| Layer | Units | Activation | Regularization |
|-------|-------|-----------|----------------|
| Dense | 128 | Swish + LeakyReLU (α=0.01) | BatchNorm, Dropout(0.2) |
| Dense | 64 | LeakyReLU (α=0.01) | BatchNorm, Dropout(0.2) |
| Dense | 32 | LeakyReLU (α=0.01) | BatchNorm |
| Dense | 1 | Sigmoid | — |

**Optimizer:** Adam (lr = 0.0005)  
**Loss:** Binary Cross-Entropy  
**Training:** 150 epochs, batch size 50, 20% validation split  
**Weight Init:** He Normal (optimal for ReLU-family activations)

### Conformal Prediction Scoring

After inference, each sample gets a **conformity score** defined as:

```
conformity_score = 1 - |predicted_probability - true_label|
```

A score close to **1.0** means the model's output strongly aligns with the ground truth — high certainty. Scores near **0.5** indicate uncertainty. This distribution is visualized in `conformity_distribution.png`.

In the Gradio UI, the **confidence** reported is the raw sigmoid output and **conformity** is `max(p, 1-p)` — a proxy for prediction certainty regardless of class.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision | Of all flagged attacks, how many are real |
| Recall / Sensitivity | Of all real attacks, how many were caught |
| F1 Score | Harmonic mean of precision and recall |
| AUC-ROC | Area under the ROC curve (classifier separability) |

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn gradio
```

Or create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn gradio
```

---

## Usage

### Step 1: Generate Dataset

**Option A — Synthetic data:**

```bash
python dataset.py
```

This creates `train.csv` and `test.csv` from scratch.

**Option B — Real sensor data:**

Place your real data in `data.csv` (features only, or with a `Label` column set to 0), then run:

```bash
python standardization.py
```

This creates `train_standardized.csv` and `test_standardized.csv`.

---

### Step 2: Train & Evaluate

```bash
python r3_gradio.py
```

This will:
1. Load `train.csv` and `test.csv`
2. Standardize features using `StandardScaler`
3. Train the DNN for 150 epochs
4. Evaluate on the test set and save a timestamped report to `model_report_<timestamp>.csv`
5. Save conformity scores to `conformity_scores_<timestamp>.csv`
6. Generate plots in the `plots/` directory
7. Launch the Gradio web interface at `http://127.0.0.1:7860`

---

### Step 3: Live Inference via Gradio

Once the Gradio UI launches in your browser, enter **5 comma-separated, standardized feature values**:

```
-0.246,-0.684,-1.826,1.057,-0.696
```

**Output format:**
```
Result: Legit Data | Confidence: 0.0412 | Conformity: 0.9588
```

- **Result:** `Legit Data` or `False Data`
- **Confidence:** Model's sigmoid output (probability of being false data)
- **Conformity:** `max(p, 1-p)` — certainty of the prediction

> **Note:** Input values must be pre-standardized (zero mean, unit variance) to match the scaler fitted during training. Use `normal_to_stan.py` or `standardization.py` as a reference.

---

## File Descriptions

| File | Purpose |
|------|---------|
| `r3_gradio.py` | Main pipeline: data loading, preprocessing, model training, evaluation, plotting, and Gradio UI |
| `dataset.py` | Generates synthetic FDI dataset from Gaussian distributions |
| `standardization.py` | Preprocesses real sensor data and injects synthetic false data |
| `normal_to_stan.py` | Standalone utility demonstrating manual z-score standardization |
| `data.csv` | Raw (real-world) sensor data used by `standardization.py` |
| `train.csv` / `test.csv` | Auto-generated training and test splits from `dataset.py` |
| `values_true.txt` | Sample standardized legitimate sensor readings for UI testing |
| `values_false.txt` | Sample anomalous/injected sensor readings for UI testing |
| `model_logs.log` | Timestamped log of all dataset loads and HTTP events |

---

## Model Performance

Results from the most recent evaluation run (`model_report_20250421-122430.csv`):

| Metric | Value |
|--------|-------|
| **Accuracy** | **94.49%** |
| **Precision** | 87.73% |
| **Recall** | 65.60% |
| **F1 Score** | 75.07% |
| **AUC-ROC** | **92.51%** |
| **Sensitivity** | 65.60% |

> The model achieves strong overall accuracy and AUC-ROC. The lower recall reflects the class imbalance (only 15% false data), meaning the model is conservative — it avoids false alarms but may miss some injections. This can be tuned via the classification threshold (currently `p > 0.5`).

---

## Sample Inputs

### Legitimate Data (from `values_true.txt`)

```
-0.2458191267110575,-0.6838602033464366,-1.8255121089847095,1.0570434191653657,-0.6958299463922752
2.1332646815618554,-1.3686710685200578,1.0254935862662529,0.4997262729846501,0.8377462255551519
```

### Injected False Data (from `values_false.txt`)

```
44.078187825905338,-0.3757890537874079,-0.2864434883174691,-1.3093278631077054,1.4624093974362882
-1.7848745879867591,1.8309596410387543,-2.417042929140234,2.4058005412679373,-0.2418456109816837
```

Note how false data often has extreme outlier values (e.g., `44.07`) characteristic of sensor tampering.

---

## Output Files

| File | Description |
|------|-------------|
| `plots/training_history.png` | Train/Val accuracy and loss per epoch |
| `plots/roc_curve.png` | ROC curve with AUC value |
| `plots/conformity_distribution.png` | Histogram of per-sample conformity scores |
| `model_report_<timestamp>.csv` | 6 evaluation metrics for the run |
| `conformity_scores_<timestamp>.csv` | Conformity score for every test sample |

---

## Logging

All events are written to `model_logs.log` in the following format:

```
2025-04-21 12:24:05,776 - INFO - Loaded dataset: train.csv
2025-04-21 12:24:31,385 - INFO - HTTP Request: GET http://127.0.0.1:7860/gradio_api/startup-events "HTTP/1.1 200 OK"
```

This includes dataset load events and Gradio HTTP request traces, useful for debugging and audit trails.

---

## Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
                         │            Smart Grid Sensors            │
                         │   (Power Flow, Voltage, Temperature...)  │
                         └─────────────────┬───────────────────────┘
                                           │
                                    Raw Readings
                                           │
                         ┌─────────────────▼───────────────────────┐
                         │          Data Preprocessing              │
                         │    StandardScaler (fit on train set)     │
                         └─────────────────┬───────────────────────┘
                                           │
                              Standardized Feature Vector
                                           │
                         ┌─────────────────▼───────────────────────┐
                         │            DNN Classifier                │
                         │  Dense(128) → Dense(64) → Dense(32)     │
                         │    BatchNorm + LeakyReLU + Dropout       │
                         │           → Dense(1, Sigmoid)           │
                         └─────────────────┬───────────────────────┘
                                           │
                              Output: P(False Data) ∈ [0, 1]
                                           │
                         ┌─────────────────▼───────────────────────┐
                         │          Conformal Scoring               │
                         │  conformity = max(p, 1-p)               │
                         └─────────────────┬───────────────────────┘
                                           │
                         ┌─────────────────▼───────────────────────┐
                         │           Gradio Web UI                  │
                         │  "Legit Data" / "False Data"            │
                         │  + Confidence + Conformity               │
                         └─────────────────────────────────────────┘
```

---

*Built as part of the False Data Injection in Smart Grids research project.*
