# coding: utf-8
from common import config
# GPU에서 실행할 경우 주석 해제 (cupy)
# ==============================================
# config.GPU = True
# ==============================================
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
from modules.better_rnnlm import BetterRnnlm
import pickle
import time as t

# 이어서 학습하기
load_params_pkl = None
load_params_pkl = 'ln_30160000 lt_53h57m54s ppl_94.7 BetterRnnlm params'
ylim = 500


# 하이퍼 파라미터 설정
wordvec_size = 500
hidden_size = 500
batch_size = 20
time_size = 40
lr = 10.0
max_epoch = 20
max_grad = 0.25
dropout = 0.5


# 학습할 데이터 읽어 학습/검증/테스트 데이터로 나누기
with open('saved_pkls/YT_cmts_morps_to_id_Kkma.pkl', 'rb') as f:
    (corpus_all, morp_to_id, id_to_morp) = pickle.load(f)

max_iter = round(len(corpus_all)*0.8/batch_size/time_size/100)*100
len_train, len_val = batch_size*time_size*max_iter+1, int(len(corpus_all)*0.1)
corpus_train = corpus_all[ :len_train]
corpus_val = corpus_all[len_train : len_train+len_val]
corpus_test = corpus_all[len_train+len_val: ]
print("\ntrain {}, val {}, test {} data".format(len(corpus_train), len(corpus_val), len(corpus_test)))

if config.GPU:
    corpus_train = to_gpu(corpus_train)
    corpus_val = to_gpu(corpus_val)
    corpus_test = to_gpu(corpus_test)

vocab_size = len(id_to_morp) ### len(morp_to_id) < len(id_to_morp)
xs = corpus_train[:-1]
ts = corpus_train[1:]


# 신경망과 훈련 모듈 만들거나 가져오기 (그래프 출력)
model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr) ### Adam -> overflow
trainer = RnnlmTrainer(model, optimizer)

if load_params_pkl != None:
    model.load_params(load_params_pkl)
    trainer.load_pplist(load_params_pkl.replace('params', 'pplist'))
    trainer.plot(ylim=ylim)
print() # len(trainer.ppl_list)


# 학습
best_ppl = float('inf')
start_time = t.time()
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
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
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)
