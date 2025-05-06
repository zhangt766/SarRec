import inspect
import torch
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
import pandas as pd

from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel


def evaluate_userwise_ranking(scored_df, prediction_set_dict):
    total_users = len(prediction_set_dict)
    hit_total = 0
    ndcg_total = 0
    mrr_total = 0

    for sid, pred_items in prediction_set_dict.items():
        group = scored_df[scored_df["sample_id"] == sid]
        if group.empty or len(pred_items) == 0:
            continue

        sub_group = group[group["candidate"].isin(pred_items)]
        sorted_group = sub_group.sort_values("prob", ascending=False).reset_index(drop=True)
        labels = sorted_group["label"].values

        if len(labels) == 0:
            continue

        # Hit
        hit = int(1 in labels)
        hit_total += hit

        # NDCG
        if 1 in labels:
            rank = np.where(labels == 1)[0][0] + 1
            dcg = 1 / np.log2(rank + 1)
            idcg = 1
            ndcg = dcg / idcg
            mrr = 1 / rank
        else:
            ndcg = 0
            mrr = 0

        ndcg_total += ndcg
        mrr_total += mrr

    return {
        "userwise_hit": hit_total / total_users if total_users > 0 else 0.0,
        "userwise_ndcg": ndcg_total / total_users if total_users > 0 else 0.0,
        "userwise_mrr": mrr_total / total_users if total_users > 0 else 0.0
    }


class MInterface(pl.LightningModule):
    def __init__(self,
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.scored_rows = []
        self.lambda_threshold = 0.05  # 默认值
        lambda_path = getattr(self.hparams, "lambda_path", "")
        if lambda_path and os.path.isfile(lambda_path):
            self.lambda_threshold = torch.load(lambda_path)["lambda"]
        self.load_llm(self.hparams.llm_path)
        self.load_rec_model(self.hparams.rec_model_path)
        self.load_projector()

    def forward(self, batch):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        # 仅在存在 token_type_ids 时才应用 mask
        if "token_type_ids" in batch["tokens"]:
            targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:, 1:], -100)

        input_embeds = self.wrap_emb(batch)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits
        }

    def generate(self, batch, temperature=0.8, do_sample=False, num_beams=1, max_gen_length=64, min_gen_length=1,
                 repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1):
        input_embeds = self.wrap_emb(batch)
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences
        )
        output_text = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
        outputs = [text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)

        if batch["flag"]:
            for param in self.projector.parameters():
                param.requires_grad = False
        else:
            for param in self.projector.parameters():
                param.requires_grad = True

        out = self(batch)
        logits = out["logits"][:, -1, :]
        probs = F.softmax(logits / self.temperature, dim=-1)

        targets = batch["item_id"]
        loss = F.cross_entropy(logits, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_scored_rows = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        logits = out["logits"][:, -1, :]
        probs = F.softmax(logits / self.temperature, dim=-1)

        for i in range(len(batch["cans_name"])):
            cans = batch["cans_name"][i]
            real = batch["correct_answer"][i]
            for can in cans:
                token_ids = self.llama_tokenizer(can, add_special_tokens=False).input_ids
                if len(token_ids) == 0:
                    continue
                token_id = token_ids[0]
                prob = probs[i, token_id].item()
                self.val_scored_rows.append({
                    "sample_id": batch_idx * len(batch["cans_name"]) + i,
                    "candidate": can,
                    "prob": prob,
                    "label": int(can.strip().lower() == real.strip().lower()),
                })

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real, cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        if len(self.val_scored_rows) == 0:
            print("⚠ No validation scores recorded.")
            return

        val_df = pd.DataFrame(self.val_scored_rows)
        metrics = evaluate_ranking_metrics(val_df, ks=[5, 10])

        for key, value in metrics.items():
            self.log(f"val_{key}", round(value, 4), on_step=False, on_epoch=True, prog_bar=True)
            print(f"val_{key}: {value:.4f}")

    def on_test_epoch_start(self):
        self.test_content = {
            "generate": [],
            "real": [],
            "cans": [],
        }
        self.scored_rows = []

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        out = self(batch)
        temperature = 1  # 可调，<1 放大差距，>1 拉平
        logits = out["logits"][:, -1, :]  # shape: [B, V]
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True)
        std[std == 0] = 1e-6  # 避免除 0
        logits = (logits - mean) / std

        probs = F.softmax(logits / temperature, dim=-1)

        for i in range(len(batch["cans_name"])):
            cans = batch["cans_name"][i]
            real = batch["correct_answer"][i]
            for can in cans:
                token_ids = self.llama_tokenizer(can, add_special_tokens=False).input_ids
                if len(token_ids) == 0:
                    continue
                token_id = token_ids[0]
                prob = probs[i, token_id].item()
                self.scored_rows.append({  # ← 注意累积保存
                    "sample_id": batch_idx * len(batch["cans_name"]) + i,  # 全局 sample_id
                    "candidate": can,
                    "prob": prob,
                    "label": int(can.strip().lower() == real.strip().lower()),
                    "real": real,
                    "generate": "N/A",
                })

        return []

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real, cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        if hasattr(self, 'scored_rows') and len(self.scored_rows) > 0:
            scored_df = pd.DataFrame(self.scored_rows)
            if not os.path.exists(self.hparams.output_dir):
                os.makedirs(self.hparams.output_dir)
            scored_df.to_csv(op.join(self.hparams.output_dir, 'test_scored.csv'), index=False)

            lambda_threshold = 0.0003  # 统一使用 class 参数
            df_filtered = scored_df[scored_df["prob"] >= lambda_threshold]
            all_sample_ids = scored_df["sample_id"].unique()

            prediction_sets = []
            for sid in all_sample_ids:
                group = df_filtered[df_filtered["sample_id"] == sid]
                candidates = group["candidate"].tolist()
                true_row = scored_df[(scored_df["sample_id"] == sid) & (scored_df["label"] == 1)]
                real_item = true_row.iloc[0]["candidate"] if not true_row.empty else None
                is_covered = int(real_item in candidates) if real_item else 0

                prediction_sets.append({
                    "sample_id": sid,
                    "prediction_set": candidates,
                    "set_size": len(candidates),
                    "real": real_item,
                    "is_covered": is_covered
                })

            pred_df = pd.DataFrame(prediction_sets)
            pred_df.to_csv(op.join(self.hparams.output_dir, "prediction_set.csv"), index=False)

            # 计算常规指标
            coverage = pred_df["is_covered"].mean()
            retrieval_valid = pred_df[pred_df["real"].notna()]
            avg_size = retrieval_valid["set_size"].mean() if len(retrieval_valid) > 0 else 0.0
            total_samples = scored_df["sample_id"].nunique()
            retrieval_hits = scored_df[scored_df["label"] == 1]["sample_id"].nunique()
            retrieval_recall = retrieval_hits / total_samples if total_samples > 0 else 0.0

            self.log("test_coverage", round(coverage, 4), prog_bar=True)
            self.log("test_avg_pred_set_size", round(avg_size, 2), prog_bar=True)
            self.log("test_retrieval_recall", round(retrieval_recall, 4), prog_bar=True)

            print(f"test_coverage: {coverage:.2f}")
            print(f"avg_pred_set_size: {avg_size:.0f}")
            print(f"retrieval_recall: {retrieval_recall:.2f}")

            # User-wise Ranking: Hit / NDCG / MRR based on dynamic prediction set
            prediction_set_dict = {
                row["sample_id"]: row["prediction_set"]
                for _, row in pred_df.iterrows()
            }
            metrics = evaluate_userwise_ranking(scored_df, prediction_set_dict)

            for key, value in metrics.items():
                self.log(f"test_{key}", round(value, 4), prog_bar=True)
                print(f"{key}: {value:.4f}")

        else:
            print("No scoring results found (self.scored_rows is empty).")

        df = DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))

        if len(self.test_content.get("generate", [])) == 0:
            print("⚠ No 'generate' content found. Skipping HR@1 computation.")
            return

        prediction_valid_ratio, hr = self.calculate_hr1(self.test_content)
        metric = hr * prediction_valid_ratio
        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            {'params': self.projector.parameters(), 'lr': self.hparams.lr, 'weight_decay': weight_decay},
            {'params': self.llama_model.parameters(), 'lr': self.hparams.lr}
        ])

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                               max_step=max_step,
                                                               min_lr=self.hparams.lr_decay_min_lr,
                                                               init_lr=self.hparams.lr,
                                                               warmup_steps=warmup_steps,
                                                               warmup_start_lr=self.hparams.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        elif self.hparams.save == 'all':
            pass

    def load_llm(self, llm_path):
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]']})
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.hparams.llm_tuning == 'lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj',
                                                             'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'freeze':
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        elif self.hparams.llm_tuning == 'freeze_lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj',
                                                             'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()
        else:
            raise NotImplementedError()

        print('Loading LLAMA Done')

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size,
                                            llm_size=self.llama_model.config.hidden_size)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        self.rec_model = torch.load(rec_model_path, map_location="cpu")
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False

    def encode_items(self, seq):
        if self.hparams.rec_embed == "SASRec":
            item_rec_embs = self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser', 'GRU']:
            item_rec_embs = self.rec_model.item_embeddings(seq)
        item_txt_embs = self.projector(item_rec_embs)
        return item_txt_embs

    def embed_tokens(self, token_ids):
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)

        his_token_id = self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",
                                            add_special_tokens=False).input_ids.item()
        cans_token_id = self.llama_tokenizer("[CansEmb]", return_tensors="pt",
                                             add_special_tokens=False).input_ids.item()
        item_token_id = self.llama_tokenizer("[ItemEmb]", return_tensors="pt",
                                             add_special_tokens=False).input_ids.item()
        his_item_embeds = self.encode_items(batch["seq"])
        cans_item_embeds = self.encode_items(batch["cans"])
        item_embeds = self.encode_items(batch["item_id"])

        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i] == his_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, :batch["len_seq"][i].item()]):
                    input_embeds[i, idx] = item_emb
            if (batch["tokens"].input_ids[i] == cans_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, cans_item_embeds[i, :batch["len_cans"][i].item()]):
                    input_embeds[i, idx] = item_emb
            if (batch["tokens"].input_ids[i] == item_token_id).nonzero().shape[0] > 0:
                idx = (batch["tokens"].input_ids[i] == item_token_id).nonzero().item()
                input_embeds[i, idx] = item_embeds[i]
        return input_embeds

