import torch
import os.path as op
import numpy as np
import torch.utils.data as data
import pandas as pd
import random

class LastfmData(data.Dataset):
    def __init__(self, data_dir=r'data/ref/lastfm',
                 stage=None,
                 cans_num=10,
                 sep=", ",
                 no_augment=True):
        self.__dict__.update(locals())
        self.aug = (stage == 'train') and not no_augment
        self.padding_item_id = 4606
        self.check_files()
        self.user_emb = np.load(op.join(self.data_dir, f'{stage}_user_embedding.npy'))
        self.item_emb = np.load(op.join(self.data_dir, 'item_embedding.npy'))
        self.tau = torch.load(op.join(self.data_dir, 'trained_tau.pth'))['tau']
        self.tau_temp = 0.1

    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]

        candidates = self.retrieve_topk(i)  # 使用 RAG 检索构造候选集
        cans_name = [self.item_id2name[can] for can in candidates]

        sample = {
            'seq': temp['seq'],
            'seq_name': temp['seq_title'],
            'len_seq': temp['len_seq'],
            'seq_str': self.sep.join(temp['seq_title']),
            'cans': candidates,
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': len(candidates),
            'item_id': temp['next'],
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        return sample

    def retrieve_topk(self, user_idx, score_threshold=0.5):
        u = torch.tensor(self.user_emb[user_idx]).to(torch.float32)        # [D]
        V = torch.tensor(self.item_emb).to(torch.float32)                 # [N, D]
        tau = torch.tensor(self.tau).to(torch.float32)

        scores = torch.sigmoid((u @ V.T - tau) / self.tau_temp)           # [N]
        indices = (scores > score_threshold).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            indices = torch.topk(scores, 1).indices
        return indices.tolist()

    def check_files(self):
        self.item_id2name = self.get_music_id2name()
        if self.stage == 'train':
            filename = "train_data.df"
        elif self.stage == 'val':
            filename = "Val_data.df"
        elif self.stage == 'test':
            filename = "Test_data.df"
        data_path = op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)

    def get_music_id2name(self):
        music_id2name = dict()
        item_path = op.join(self.data_dir, 'id2name.txt')
        with open(item_path, 'r') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                music_id2name[int(ll[0])] = ll[1].strip()
        return music_id2name

    def session_data4frame(self, datapath, music_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]

        def remove_padding(xx):
            x = xx[:]
            for _ in range(10):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x

        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        train_data['seq_title'] = train_data['seq_unpad'].apply(lambda x: [music_id2name[x_i] for x_i in x])
        train_data['next_item_name'] = train_data['next'].apply(lambda x: music_id2name[x])
        return train_data
