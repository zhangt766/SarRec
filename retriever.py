# retriever.py
import torch
import torch.nn as nn
import numpy as np

class RetrieverHead(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512, tau_temp=0.1, epsilon=1e-5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        self.tau_retriever = nn.Parameter(torch.tensor(0.0))
        self.tau_temp = tau_temp
        self.epsilon = epsilon

    def forward(self, e_u, cand_emb, temp=0.05):
        B, N, D = cand_emb.size()
        u = e_u.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([u, cand_emb], dim=-1)
        s = self.mlp(x).squeeze(-1)
        w = torch.sigmoid((s - self.tau_retriever) / self.tau_temp)
        w = torch.clamp(w, min=self.epsilon, max=1.0)
        return s, w

def evaluate_retriever(retriever, session_data, user_ids, item_emb, idx2item, temp=0.05, sample_n=5000):
    retriever.eval()
    device = next(retriever.parameters()).device
    total = min(sample_n, len(session_data))
    hits = 0
    item_emb_gpu = item_emb.cuda().unsqueeze(0)

    top_k_items = {}
    k_values = []

    with torch.no_grad():
        for idx, (hist_idxs, pos_idx) in enumerate(session_data[:total]):
            hist_vecs = item_emb[hist_idxs].cuda()
            e_u = hist_vecs.mean(dim=0, keepdim=True)
            s, w = retriever(e_u, item_emb_gpu, temp=temp)
            w = w.squeeze(0)
            selected_indices = torch.where(w > 0.5)[0]
            selected_indices = selected_indices.cpu().numpy()

            if len(selected_indices) == 0:
                selected_indices = torch.topk(s.squeeze(0), 5).indices.cpu().numpy()

            k = len(selected_indices)
            k_values.append(k)

            topk_items = [idx2item[str(idx)] for idx in selected_indices]
            user_id = user_ids[idx]
            top_k_items[user_id] = topk_items

            topk = torch.topk(s.squeeze(0), min(10, k)).indices.cpu().numpy()
            if pos_idx in topk:
                hits += 1

    print(f"k 值分布：平均={np.mean(k_values):.2f}, 最小={min(k_values)}, 最大={max(k_values)}")

    with open("/content/embeddings/top_k_items_retriever.json", "w") as f:
        json.dump(top_k_items, f)
    print("top-k 物品已保存到 /content/embeddings/top_k_items_retriever.json")

    return hits / total
