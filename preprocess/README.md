# MovieLens Data Preprocessing

This directory provides scripts for preprocessing the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) and generating user-item interaction embeddings for recommendation tasks.

## 📄 Files

- **`MovieLens.ipynb`**: Builds the raw interaction dataset from the original MovieLens files.  
- **`evaluate_foldout.py`**: Verifies the integrity and basic statistics of the raw dataset.  
- **`Interaction_emb.py`**: Generates interaction-based embeddings from user-item sequences.  
- **`LightGCN.py`**: Produces collaborative embeddings using the LightGCN algorithm.  
- **`load_data.py`**: Loads processed data and embeddings for model input.




