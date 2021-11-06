# coding: utf-8
from common.config import GPU
# GPU에서 실행할 경우 주석 해제 (cupy)
# ==============================================
# GPU = True
# ==============================================
from common.optimizer import optimizers
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
from modules.better_rnnlm import BetterRnnlm
import pickle
import time as t
import argparse
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", default="saved_pkls/YT_cmts_211101~06_vocab_corpus.pkl", type=str, required=False,
                    help="path of .pkl file you can got after running 2_preprocess.py")
parser.add_argument("--save_dir", default="saved_models", type=str, required=False,
                    help="to save model directory")
parser.add_argument("--load_model", default=None, type=str, required=False,
                    help="path of model (.pkl)")
parser.add_argument("--config", default="config_Rnnlm.json", type=str, required=False,
                    help="config file")
parser.add_argument("--batch_size", default=32, type=int, required=False,
                    help="batch_size")
parser.add_argument("--time_size", default=50, type=int, required=False,
                    help="sequence size")
parser.add_argument("--lr", default=10.0, type=float, required=False,
                    help="learning rate")
parser.add_argument("--max_epoch", default=20, type=int, required=False,
                    help="max_epoch")
parser.add_argument("--max_grad", default=0.25, type=float, required=False,
                    help="max_grad")
parser.add_argument("--optimizer", default='sgd', type=str, required=False,
                    help="sgd | momentum | nesterov | adagrad | rmsprop | adam ")
args = parser.parse_args()


# 이어서 학습하기
load_model = args.load_model
pkl_dir = args.pkl_dir
ylim = 500


# 하이퍼 파라미터 설정
batch_size = args.batch_size # 20
time_size = args.time_size # 40
lr = args.lr # 10.0
max_epoch = args.max_epoch # 20
max_grad = args.max_grad # 0.25


# 학습할 데이터 읽어 학습/검증/테스트 데이터로 나누기
with open(args.data_file, 'rb') as f:
    (vocab, corpus_all, _) = pickle.load(f).values()
print("학습 데이터 로드 성공!")

max_iter = round(len(corpus_all)*0.8/batch_size/time_size/100)*100
len_train, len_val = batch_size*time_size*max_iter+1, int(len(corpus_all)*0.1)
corpus_train = corpus_all[ :len_train]
corpus_val = corpus_all[len_train : len_train+len_val]
corpus_test = corpus_all[len_train+len_val: ]
print("\ntrain {}, val {}, test {} data".format(len(corpus_train), len(corpus_val), len(corpus_test)))

if GPU:
    corpus_train = to_gpu(corpus_train)
    corpus_val = to_gpu(corpus_val)
    corpus_test = to_gpu(corpus_test)

xs = corpus_train[:-1]
ts = corpus_train[1:]

# 신경망과 훈련 모듈 만들거나 가져오기 (그래프 출력)
config = Config.load(args.config)
config.vocab_size = vocab.n_morps
print("형태소 사전 단어 수:", vocab.n_morps)

model = BetterRnnlm(config) 
optimizer = optimizers[args.optimizer.lower()](lr) ### Adam -> lr: 0.0001
trainer = RnnlmTrainer(model, optimizer)

if load_model != None:
    model.load_params(load_model, pkl_dir)
    trainer.load_pplist(load_model.replace('params', 'pplist'))
    trainer.plot(ylim=ylim)
print() # len(trainer.ppl_list)


# 학습
best_ppl = float('inf')
start_time = t.time()
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, pkl_dir=pkl_dir,
                time_size=time_size, max_grad=max_grad, verbose=False, start_time=start_time)

    model.reset_state()
    ppl = eval_perplexity(model, corpus_val, batch_size, time_size)
    print('valid perplexity: ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
    else:
        lr /= 4.0
        optimizer.lr = lr
        print('lr: ', optimizer.lr)

    model.reset_state()
    print('-' * 50)


# 테스트
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test, batch_size, time_size)
print('test perplexity: ', ppl_test)
