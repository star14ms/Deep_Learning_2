# coding: utf-8
import sys
sys.path.append('..')
import os
import pickle
from common.np import *
from common.util import to_gpu, to_cpu, time
from common.config import GPU

class BaseModel:
    def __init__(self): ### 자신을 상속하면 실행 안 됨
        self.params, self.grads = None, None 

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, str_learning_time, ppl, pkl_dir='saved_pkls'):
        file_name = f'ln_{self.learning_num} ' + f'lt_{str_learning_time} ' + \
            f'ppl_{ppl} ' + self.__class__.__name__ + ' params.pkl'

        params = [p.astype(np.float16) for p in self.params]
        if GPU:
            params = [to_cpu(p) for p in params]

        with open(pkl_dir+'/'+file_name, 'wb') as f:
            pickle.dump(params, f)
        
        print("네트워크 저장 성공!")

    def load_params(self, file_name, pkl_dir='saved_pkls'):
        
        splited_name = file_name.split(' ')
        self.learning_num = int(splited_name[0].lstrip('ln_'))
        hms = list(map(int, splited_name[1]
        .replace('lt_', '').replace('h', ' ').replace('m', ' ').replace('s', '').split(' '))) ### int
        self.learning_time = hms[0]*3600 + hms[1]*60 + hms[2]

        # if '/' in file_name:
        #     file_name = file_name.replace('/', os.sep)

        if splited_name[-1] != 'params.pkl':
            file_name = file_name+'.pkl'

        if not os.path.exists(pkl_dir+'/'+file_name):
            raise IOError('No file: ' + file_name)

        with open(pkl_dir+'/'+file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
            
        self.loaded = True
        self.load_pkl_name = pkl_dir+'/'+file_name
        print("네트워크 불러오기 성공!")
