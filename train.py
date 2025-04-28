# sarrec_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SarRecModel(nn.Module):
    def __init__(self, retriever, generator, tokenizer, item_emb, idx2item, item2idx, num_items, K_neg=200, pool_factor=5, temp=0.05, eta=1e-3):
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.tokenizer = tokenizer
        self.item_emb = item_emb
        self.idx2item = idx2item
        self.item2idx = item2idx
        self.num_items = num_items
        self.K_neg = K_neg
        self.pool_factor = pool_factor
        self.temp = temp
        self.eta = eta

    def forward(self, hist_batch, pos_batch):
        B = len(hist_batch)
        e_us = []
        for hist in hist_batch:
            e_us.append(self.item_emb[hist].mean(dim=0))
        e_us = torch.stack(e_us, dim=0).cuda()

        pos_embs = self.item_emb[pos_batch].cuda()

        pool_size = self.K_neg * self.pool_factor
        pool_idxs = np.random.choice(self.num_items, pool_size, replace=False)
        pool_embs = self.item_emb[pool_idxs].cuda().unsqueeze(0).expand(B, -1, -1)
        with torch.no_grad():
            s_pool, _ = self.retriever(e_us, pool_embs, temp=self.temp)
        topk = s_pool.topk(self.K_neg, dim=1).indices
        pool_idxs_t = torch.tensor(pool_idxs, device=topk.device)
        neg_idxs = pool_idxs_t[topk]
        neg_embs = self.item_emb[neg_idxs.cpu()].cuda()

        cand_emb = torch.cat([pos_embs.unsqueeze(1), neg_embs], dim=1)
        cand_items = [torch.tensor([pos_batch[b].item()] + neg_idxs[b].cpu().numpy().tolist()) for b in range(B)]

        s, w = self.retriever(e_us, cand_emb, temp=self.temp)
        labels = torch.zeros(B, dtype=torch.long, device=s.device)
        loss_nce = F.cross_entropy(s / self.temp, labels)
        loss_reg = self.eta * w.sum(dim=1).mean()

        prompts = []
        for b in range(B):
            hist = [self.idx2item[str(idx)] for idx in hist_batch[b]]
            cand_list = [f"- {self.idx2item[str(idx)]} (score: {score:.2f})" for idx, score in zip(cand_items[b], s[b].cpu().numpy())]
            prompt = (
                f"User has interacted with: {', '.join(map(str, hist))}.\n"
                f"Candidate items with relevance:\n"
                f"{'\n'.join(cand_list)}\n"
                f"What is the most likely next item?"
            )
            prompts.append(prompt)

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        labels = self.tokenizer([str(pos_batch[b].item()) for b in range(B)], return_tensors="pt", padding=True).input_ids.to("cuda")
        outputs = self.generator(**inputs, labels=labels)
        loss_gen = outputs.loss

        loss = loss_nce + loss_reg + loss_gen
        return loss
