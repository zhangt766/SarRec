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
preprocess files contains codes to process raw datand embeddings for model input
```

# 📦 Data Resources for MovieLens Project

All data files used in this project are hosted on Hugging Face and can be accessed at:

🔗 **[https://huggingface.co/datasets/zhangt766/Movielens/tree/main](https://huggingface.co/datasets/zhangt766/Movielens/tree/main)**

## 📁 Contents

- **Raw Data**
  - Preprocessed user-item interaction sessions
  - Original MovieLens item metadata

- **Item Embeddings**
  - Text-based or metadata-derived item vectors

- **User-Item Interaction Embeddings**
  - Interaction-based co-occurrence embeddings
  - LightGCN collaborative filtering embeddings

- **Model Checkpoints**
  - Trained weights and intermediate results for reproduction

## 📥 Usage

You can directly download files via `wget`, `curl`, or using the 🤗 Datasets Hub interface:

```bash
wget https://huggingface.co/datasets/zhangt766/Movielens/resolve/main/<filename>




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
