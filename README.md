# SarRec: Statistically‑Guaranteed Augmented Retrieval for Recommendation

## Overview
SarRec implements a retrieval‑augmented generation framework for sequential recommendation with **end‑to‑end differentiable retrieval** and **post‑hoc risk calibration**. It jointly trains a retriever and a large language model (LLM) generator under a unified objective, then applies conformal risk‑controlling calibration to deliver set‑valued recommendations with formal statistical guarantees.

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

