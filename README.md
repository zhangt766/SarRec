# SarRec: Statistically-Guaranteed Augmented Retrieval for Recommendation

## Overview
SarRec implements a retrieval-augmented generation framework for sequential recommendation with **end-to-end differentiable retrieval** and **post-hoc risk calibration**. It jointly trains a retriever and a large language model (LLM) generator under a unified objective, then applies conformal risk-controlling calibration to deliver set-valued recommendations with formal statistical guarantees.

## Installation
```bash
git clone https://github.com/zhangt766/SarRec.git
cd SarRec
pip install -r requirements.txt
```

## Data Preparation
```
python scripts/preprocess_data.py \
  --raw_dir data/raw/ \
  --out_dir data/processed/
```

## Usage
### Train retriever + generator
```bash
python src/train.py --config configs/default.yaml
```

### Calibrate risk threshold
```bash
python src/calibrate.py --config configs/default.yaml
```

### Inference with λ*
```bash
python src/inference.py --config configs/default.yaml
```
