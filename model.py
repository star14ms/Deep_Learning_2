import torch
import torch.nn as nn
import torch.nn.functional as F
import json

            
class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        v_size = config.vocab_size
        w_size = config.wordvec_size
        h_size = config.hidden_size
        n_layers = config.n_layers
        drop_ratio = config.dropout
        bidirectional = config.bidirectional

        self.Emb = nn.Embedding(v_size, w_size)
        self.Emb_drop = nn.Dropout(drop_ratio)
        self.LSTM = nn.LSTM(w_size, h_size, num_layers=n_layers, dropout=drop_ratio, bidirectional=bidirectional)
        self.LSTM_drop = nn.Dropout(drop_ratio)
        self.Liner = nn.Linear(h_size*2 if bidirectional else h_size, v_size)

    def forward(self, x):
        y = self.Emb(x)
        y = self.Emb_drop(y)
        y, _ = self.LSTM(y) # return y, (h, c)
        y = self.LSTM_drop(y)
        y = self.Liner(y)

        return y

    def save(self, epoch, ppl, path):
        torch.save({
        "epoch": epoch,
        "ppl": ppl,
        "state_dict": self.state_dict()
    }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["ppl"]

    def generate(self, start_id, skip_ids=None, sample_size=100, one_sentence=False, id2morp=None, end='.'):
        word_ids = [start_id]
        
        x = start_id
        while len(word_ids) < sample_size:
            x = torch.tensor([x]).reshape(1, 1)
            score = self.forward(x).flatten()
            p = F.softmax(score, dim=0).flatten()

            sampled = p.multinomial(num_samples=1, replacement=True)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

            if one_sentence and id2morp is not None:
                if id2morp[int(x)] == end:
                    return word_ids
                else:
                    continue

        return word_ids


""" configuration json을 읽어들이는 class """
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)