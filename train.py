from model import LSTM, Config
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

from modules.preprocess import MAX_LENGTH
from common.util import eval_perplexity, time
from common.print import add_spaces_until_endline as line
import time as t
import argparse

################################################################################################################################

start_time = t.time()

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_LSTM.json", type=str, required=False,
                    help="config file")
parser.add_argument("--data_file", default="saved_pkls/YT_cmts_211101~06_vocab_corpus.pkl", type=str, required=False,
                    help="path of .pkl file you can got after running 2_preprocess.py")
parser.add_argument("--epoch", default=3, type=int, required=False,
                    help="epoch")
parser.add_argument("--batch", default=32, type=int, required=False,
                    help="batch")
parser.add_argument("--load_model", default=None, type=str, required=False,
                    help="path of trained model (.pth)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

config = args.config
data_file = args.data_file
n_epoch = args.epoch
n_batch = args.batch

config = Config.load(config)
n_layers = config.n_layers
time_size = MAX_LENGTH

################################################################################################################################

with open(data_file, 'rb') as f:
    (vocab, corpus, _) = pickle.load(f).values()
print("학습 데이터 로드 성공!")

print("\n형태소 사전 단어 수:", vocab.n_morps)
config.vocab_size = vocab.n_morps


def split_data(corpus, ratio=[0.8, 0.1, 0.1]):
    """학습할 데이터 읽어 학습/검증/테스트 데이터로 나누기"""
    max_iter = round(len(corpus)*ratio[0]/n_batch/n_layers/100)*100
    len_train, len_val = n_batch*n_layers*max_iter+1, int(len(corpus)*ratio[1])
    train_data = corpus[ :len_train]
    val_data = corpus[len_train : len_train+len_val]
    test_data = corpus[len_train+len_val: ]
    print("\ntrain {}, val {}, test {} morps".format(len(train_data), len(val_data), len(test_data)))
    
    return train_data, val_data, test_data


class Trainer:
    def __init__(self, model, train_data, loss_fn, optimizer, n_batch, time_size, device, start_time):
        self.model = model.to(device)
        self.xs, self.ts = train_data[:-1], train_data[1:]
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n_batch = n_batch
        self.time_size = time_size
        self.device = device
        self.start_time = start_time

        self.time_idx = 0
        self.ppl_list = []

    def get_batch(self, stay_time_idx=False):
        batch_x = torch.zeros((self.n_batch, self.time_size), dtype=torch.long).to(self.device)
        batch_t = torch.zeros((self.n_batch, self.time_size), dtype=torch.long).to(self.device)
    
        data_size = len(self.xs)
        jump = data_size // self.n_batch
        offsets = [i * jump for i in range(self.n_batch)]  # バッチの各サンプルの読み込み開始位置
    
        for time in range(self.time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = self.xs[(offset + self.time_idx) % data_size]
                batch_t[i, time] = self.ts[(offset + self.time_idx) % data_size]
            self.time_idx += 1
            if offsets[-1] + self.time_idx == len(self.xs): self.time_idx = 0
        # print(offsets[-1] + self.time_idx, len(self.xs))
        
        if stay_time_idx: self.time_idx -= self.time_size
        return batch_x, batch_t

    def train_epoch(self, print_every=10):
        self.model.train()
        total_loss, local_loss = 0, 0
        max_iters  = len(self.xs) // (self.n_batch * self.time_size)

        for iter in tqdm(range(1, max_iters+1), desc=""):
            X, T = self.get_batch()
    
            # 예측 오류 계산 X: [32, 50] pred: [32, 50, 47878] y: [32, 50]
            Y = self.model(X)
            loss = self.loss_fn(Y.view(-1, Y.shape[-1]), T.view(-1))
            total_loss += loss
            local_loss += loss
    
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            if iter % print_every == 0:
                ppl = torch.exp(torch.tensor([local_loss / print_every])).item()
                elapsed = time.str_delta(self.start_time, join=':')
                string = f'\r{elapsed} | iter {iter}/{max_iters} | ppl %.1f' % ppl
                print(line(string))
                trainer.ppl_list.append(float(ppl))
                local_loss = 0

        return torch.exp(torch.tensor([total_loss / max_iters])).item()


train_data, val_data, test_data = split_data(corpus)

model = LSTM(config)
last_epoch = 0 if not args.load_model else model.load(args.load_model)[0]
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-0, momentum=0.9, nesterov=True)
trainer = Trainer(model, train_data, loss_fn, optimizer, n_batch, time_size, device, start_time)

print('-' * 50)
for epoch in range(last_epoch+1, last_epoch+n_epoch+1):
    print(f'Epoch {epoch}')

    ppl = trainer.train_epoch()
    print('train perplexity: ', ppl) # perplexity: 다음에 올 단어 후보 수

    model.eval()
    with torch.no_grad():
        ppl = eval_perplexity(model, val_data, n_batch, time_size, loss_fn, use_torch=True, data_type="valid")
        
    model.save(epoch, ppl, f"saved_modles/{model.__class__.__name__} ep_{epoch} ppl_%.1f.pth" % ppl)
    print('-' * 50)

with torch.no_grad():
    ppl = eval_perplexity(model, test_data, n_batch, time_size, loss_fn, use_torch=True, data_type="test")
