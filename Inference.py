
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever import RetrieverHead, evaluate_retriever

# 加载 idx2item 和 session_data
with open('/content/idx2item.json', 'r') as f:
    idx2item = json.load(f)
item2idx = {int(v): int(k) for k, v in idx2item.items()}

session_data = []
user_ids = []
with open('/content/train.jsonl', 'r', encoding='utf-8') as fin:
    for line in fin:
        rec = json.loads(line.strip())
        raw_hist = rec.get('history', [])
        raw_nxt = rec.get('label')
        hist_idx = [item2idx[mid] for mid in raw_hist]
        nxt_idx = item2idx[raw_nxt]
        session_data.append((hist_idx, nxt_idx))
        user_ids.append(str(rec["user_id"]))

# 加载物品嵌入
item_emb_path = '/content/concat_item_emb.npy'
item_emb_np = np.load(item_emb_path)
item_emb = torch.from_numpy(item_emb_np).float().cuda()
num_items = item_emb.size(0)
emb_dim = item_emb_np.shape[1]

# 加载微调后的模型
OUTPUT_DIR = "lora_llama/Llama-3.1-8B-Instruct_mixed_text_sarrec"
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# 加载检索器
retriever = RetrieverHead(emb_dim, tau_temp=0.1, epsilon=1e-5).cuda()
retriever.load_state_dict(torch.load(f"{OUTPUT_DIR}/retriever.pth"))

# 动态 top-k 检索
evaluate_retriever(retriever, session_data, user_ids, item_emb, idx2item, temp=0.05)

# 构造提示
with open("/content/train.jsonl", "r") as f:
    user_data = {}
    for line in f:
        data = json.loads(line)
        user_id = str(data["user_id"])
        user_data[user_id] = data

with open("/content/embeddings/top_k_items_retriever.json", "r") as f:
    top_k_items = json.load(f)

prompts = {}
for user_id, data in user_data.items():
    if user_id not in top_k_items:
        continue
    new_history = top_k_items[user_id]
    prompt = f"User has interacted with: {', '.join(map(str, new_history))}.\nWhat is the most likely next item?"
    prompts[user_id] = prompt

with open("/content/sarrec_prompts.json", "w") as f:
    json.dump(prompts, f)
print("提示已保存到 /content/sarrec_prompts.json")

# 生成推荐
predictions = {}
model.eval()
with torch.no_grad():
    for user_id, prompt in prompts.items():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions[user_id] = response

with open("/content/sarrec_predictions.json", "w") as f:
    json.dump(predictions, f)
print("推荐结果已保存到 /content/sarrec_predictions.json")
