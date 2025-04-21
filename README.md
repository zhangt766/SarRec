# SarRec

**Statistically‑Guaranteed Augmented Retrieval for Sequential Recommendation**

## Overview

SarRec implements a retrieval‑augmented generation framework for sequential recommendation with **end‑to‑end differentiable retrieval** and **post‑hoc risk calibration**. It jointly trains a retriever and a large language model (LLM) generator under a unified objective, then applies conformal risk‑controlling calibration to deliver set‑valued recommendations with formal statistical guarantees.

---

## 🚀 Features

- **Differentiable Threshold Retrieval**  
  Soft, learnable retrieval threshold → smooth gradient flow between retriever and LLM.
- **Joint Training Objective**  
  Generator log‑likelihood + retrieval set‑size penalty → adaptive, user‑aware recall.
- **Post‑Hoc Calibration**  
  Conformal risk‑controlling on a held‑out calibration set → guarantees on prediction risk/coverage.
- **LoRA + 8‑bit Quantization**  
  Efficient fine‑tuning of LLMs (e.g. Llama‑3.1) with minimal GPU memory footprint.
- **Modular Codebase**  
  Clear separation: retrieval, generation, training, calibration, inference.

---

## 📁 Repository Structure


## Installation
git clone https://github.com/your-org/SarRec.git
cd SarRec
pip install -r requirements.txt

## Data Preparation
python scripts/preprocess_data.py \
  --raw_dir data/raw/ \
  --out_dir data/processed/

## Usage
### 1. Train retriever + generator
python src/train.py --config configs/default.yaml

### 2. Calibrate risk threshold
python src/calibrate.py --config configs/default.yaml

### 3. Inference with λ*
python src/inference.py --config configs/default.yaml

