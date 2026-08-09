# 🚗 EDR Forensic Reconstruction & Crash Severity Prediction System

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](requirements.txt)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](onnx_models/)

An industrial-grade forensic machine learning & deep learning system designed to reconstruct automotive crash severity (`Deploy`, `Near-Deploy`, `Non-Deploy`) using pre-impact telemetry logged across **46,533 NHTSA Event Data Recorder (EDR) records (2017–2024)**.

---

## 📌 Executive Summary & Key Highlights

- **🔒 Zero Data Leakage Protocol:** Features a strict Train-First MICE Imputation pipeline (`IterativeImputer` with `BayesianRidge`), fitting on 80% Training set ONLY to transform 20% Test data cleanly.
- **⚖️ Imbalanced Class Learning (3 Pillars):** Combines SMOTE oversampling ($k=3$), cost-sensitive tree class weighting (`class_weight='balanced'`), and PyTorch Focal Loss ($\gamma=2.0$).
- **🏆 Multi-Model Benchmark Leaderboard:**
  - 🥇 **XGBoost Classifier:** **0.8832 Macro F1** | **97.28% Accuracy** | **0.0502 MAE**
  - 🥈 **LightGBM Classifier:** **0.8674 Macro F1** | **0.9932 AUC-ROC** | SHAP Interpretability
  - 🤖 **FT-Transformer (PyTorch):** **65.82% Near-Deploy Recall** *(Highest minority sensitivity)*
- **📦 ONNX Cross-Platform Runtime:** All 6 trained models are exported to standard Open Neural Network Exchange (ONNX) format with **100.00% prediction class parity** against native format binaries.
- **🛡️ Physical NHTSA Simulation Validation:** Validated against 4 real-world NHTSA crash scenarios with **100% empirical ground truth agreement**.

---

## 📊 Performance Benchmark Leaderboard

| Model Architecture | Macro F1 (Primary) | Near-Deploy Recall | Accuracy | MAE (Lower is Better) | RMSE | $R^2$ Score | AUC-ROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ⚡ **XGBoost Classifier** | **0.8832** | **60.76%** | **97.28%** | **0.0502** | **0.3101** | **0.8848** | **0.9918** |
| 🟢 **LightGBM Classifier** | **0.8674** | **62.03%** | **97.14%** | **0.0519** | **0.3139** | **0.8820** | **0.9932** |
| 🌲 **Random Forest** | **0.8646** | **58.23%** | **96.77%** | **0.0597** | **0.3384** | **0.8628** | **0.9869** |
| 🐱 **CatBoost Classifier** | **0.8562** | **60.76%** | **97.18%** | **0.0503** | **0.3075** | **0.8867** | **0.9921** |
| 🤖 **FT-Transformer** *(PyTorch)* | **0.8166** | **65.82%** ⭐ | **96.60%** | **0.0576** | **0.3243** | **0.8740** | **0.9760** |
| 🪽 **SAINT Transformer** *(PyTorch)* | **0.7942** | **62.03%** | **94.28%** | **0.1040** | **0.4446** | **0.7632** | **0.9757** |

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/Byakuya3456/EDR_Project.git
cd EDR_Project
pip install -r requirements.txt
```

### 2. Execution & Notebook Inspection
Open and run the primary executed notebook:
```bash
jupyter lab EDR_Pipeline_Executed.ipynb
```

### 3. ONNX Runtime Inference Example
Run inference on saved ONNX models in Python, C++, or C#:
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("onnx_models/xgboost_model.onnx")
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name: sample_features.astype(np.float32)})
```

---

## 📑 How to Cite This Work

If you reference, view, or cite this codebase, ONNX models, dataset pipeline, or forensic reconstruction methodology in academic research, papers, or projects, **you MUST provide proper attribution**:

```bibtex
@misc{srijith2026edr,
  author = {SRIJITH},
  title = {EDR Forensic Reconstruction & Crash Severity Prediction System},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Byakuya3456/EDR_Project}},
  note = {Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)}
}
```

---

## 📁 Repository Structure

```
EDR_Project/
├── LICENSE                                # CC BY-NC-ND 4.0 Public License
├── README.md                              # Main Project Overview & Benchmark Documentation
├── requirements.txt                       # Python Dependencies Manifest
├── EDR_Pipeline.ipynb                     # Clean Primary Pipeline Notebook Scaffold
├── EDR_Pipeline_Executed.ipynb            # Fully Executed Notebook with Output Renderings
├── edr_pipeline_output/                   # Fused Master Dataset (46,533 records) & High-Res Graphics
├── onnx_models/                           # All 6 Exported ONNX Format Models & Parity Verification Report
├── project_report/                        # Detailed Research Reports & Metric Summaries
└── saved_models/                          # Native Serialized Model Weights & Preprocessors
```

---

## 📜 License & Copyright Protection

This work is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)**.

To view a copy of this license, visit [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/).

### Summary of Legal Permissions:
- 👤 **Attribution Required:** You MUST give appropriate credit to **SRIJITH**, provide a link to the license, and indicate if changes were made.
- 🚫 **NonCommercial:** You MAY NOT use the material for commercial purposes or monetary gain.
- 🚫 **NoDerivatives:** If you remix, transform, or build upon the material, you MAY NOT distribute the modified material.

Copyright (c) 2026 **SRIJITH**. All Rights Reserved.
