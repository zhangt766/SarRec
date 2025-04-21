# SarRec

**Statistically‑Guaranteed Augmented Retrieval for Sequential Recommendation**

## Overview
SarRec jointly trains a differentiable retriever and an LLM generator (with LoRA + 8‑bit quant) under a unified loss, then applies post‑hoc calibration to deliver recommendation sets with formal risk guarantees.

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

