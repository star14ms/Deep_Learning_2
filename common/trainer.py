# coding: utf-8
import pickle
import sys
sys.path.append('..')
import numpy
import time as t
import matplotlib.pyplot as plt
from common.np import *  # import numpy as np
from common.util import clip_grads, time
import pickle, os

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = t.time()
        for epoch in range(max_epoch):
            # シャッフル
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.str_hms_delta(start_time)
                    print('%s | epoch %d | iter %d/%d | loss %.3f'
                          % (elapsed_time, self.current_epoch+1, iters+1, max_iters, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        plt.ylim(0, max(self.ppl_list) if ylim==None else ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = []
        self.eval_interval = None
        self.batch_size = None
        self.time_size = None
        self.pre_saved_pkl = None
        self.loaded = False

    def get_batch(self, x, t, batch_size, time_size, stay_time_idx=False):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
            if offsets[-1] + self.time_idx == len(x): self.time_idx = 0
        # print(offsets[-1] + self.time_idx, len(x))

        if stay_time_idx: self.time_idx -= time_size
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=10, save_per_eval=10, verbose=False, start_time=None):
        
        if start_time==None: start_time = t.time()
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        eval_num = 0
        self.save_per_eval = save_per_eval

        if self.eval_interval != None:
            batch_size = self.batch_size
            time_size = self.time_size
            eval_interval = self.eval_interval
            if self.pre_saved_pkl == None: 
                self.pre_saved_pkl = model.load_pkl_name
        else:
            self.batch_size = batch_size
            self.time_size = time_size
            self.eval_interval = eval_interval
            self.time_idx = 0
           
        # 처음 perplexity 평가
        batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size, stay_time_idx=True)
        
        elapsed_time = time.str_hms_delta(start_time, join=':')
        print('{} | learned {} | iter {}/{} | ppl None'
              .format(elapsed_time, model.learning_num, self.time_idx//self.time_size, max_iters))

        for epoch in range(max_epoch):
            for iters in range(self.time_idx//self.time_size, max_iters):
    
                # 데이터를 미니배치로 만들어 손실, 기울기 구하고 매개변수 갱신
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
                model.learning_num += batch_size * time_size
                if verbose: print("loss: {}".format(round(loss, 3)))

                if (eval_interval is not None) and ((iters+1) % eval_interval) == 0: ### iters+1 % 
                    
                    # perplexity 평가
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.str_hms_delta(start_time, join=':')
                    print('{} | learned {} | iter {}/{} | ppl {}'
                          .format(elapsed_time, model.learning_num, iters+1, max_iters, round(ppl, 1)))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

                    # 모델, perplexity의 변화 저장
                    eval_num += 1
                    if eval_num % save_per_eval == 0: ### iters+1 %
                        learning_time = model.learning_time + t.time()-start_time
                        str_learning_time = time.str_hms(learning_time, hms=True, join='')
                        model.save_params(str_learning_time, round(ppl, 1))
                        self.save_pplist(str_learning_time, round(ppl, 1))
                        eval_num = 0
                        print()

    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        plt.ylim(0, max(self.ppl_list) if ylim==None else ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('learning_num (x' + str(self.batch_size * self.time_size * self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()
    
    def save_pplist(self, str_learning_time, ppl, delete_pre_save=True, pkl_dir='saved_pkls'):
        file_name = f'ln_{self.model.learning_num} ' + f'lt_{str_learning_time} ' + \
            f'ppl_{ppl} ' + self.model.__class__.__name__ + ' pplist.pkl'
        
        pplist_data = {
            'ppl_list': self.ppl_list, 
            'batch_size': self.batch_size,
            'time_size': self.time_size,
            'eval_interval': self.eval_interval,
            'time_idx': self.time_idx,
            'lr': self.optimizer.lr,
            }
        
        with open(pkl_dir+'/'+file_name, 'wb') as f:
            pickle.dump(pplist_data, f)

        if delete_pre_save: self.delete_pkl(remove_pplist=True)
        self.pre_saved_pkl = (pkl_dir+'/'+file_name).replace('pplist', 'params')

        print("ppl list 저장 성공!")

    def load_pplist(self, file_name, pkl_dir='saved_pkls'):
        
        if file_name.split(' ')[-1] != 'pplist.pkl':
            file_name = file_name+'.pkl'
        
        if not os.path.exists(pkl_dir+'/'+file_name):
            raise IOError('No file: ' + file_name)

        with open(pkl_dir+'/'+file_name, 'rb') as f:
            load_data = pickle.load(f)

        self.ppl_list      = load_data['ppl_list']
        self.batch_size    = load_data['batch_size']
        self.time_size     = load_data['time_size']
        self.eval_interval = load_data['eval_interval']
        self.time_idx      = load_data['time_idx']
        # self.optimizer.lr  = load_data['lr']

        print("ppl list 불러오기 성공!")

    def delete_pkl(self, real_save_per=10, remove_pplist=True, pkl_dir='saved_pkls'):
        if self.pre_saved_pkl is None: return
        pre_saved_pkl_ln = int(self.pre_saved_pkl.split(' ')[0].lstrip(pkl_dir+'/ln_'))
        
        save_per_ln = self.batch_size * self.time_size * self.eval_interval * self.save_per_eval * real_save_per
        # print(pre_saved_pkl_ln, save_per_ln, os.path.isfile(self.pre_saved_pkl), self.pre_saved_pkl)
        if os.path.isfile(self.pre_saved_pkl) and pre_saved_pkl_ln % save_per_ln != 0:
            os.remove(self.pre_saved_pkl)

        if remove_pplist:
            file_path2 = self.pre_saved_pkl.replace('params', 'pplist')
            if os.path.isfile(file_path2): os.remove(file_path2)

def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
