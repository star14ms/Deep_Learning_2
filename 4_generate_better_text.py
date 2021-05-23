# coding: utf-8
from common.np import *
from modules.rnnlm_gen import BetterRnnlmGen
import pickle
from konlpy.tag import Kkma
from common.util import generate_words


##### 변수 선언 #########################################################################################################


load_model_pkl = 'ln_30160000 lt_53h57m54s ppl_94.7 BetterRnnlm params'
morp_to_id_pkl = 'saved_pkls/YT_cmts_morps_to_id_Kkma.pkl'

with open(morp_to_id_pkl, 'rb') as f:
    (_, morp_to_id, id_to_morp) = pickle.load(f)
vocab_size = len(id_to_morp)

model = BetterRnnlmGen(vocab_size)
model.load_params(load_model_pkl)
kkma = Kkma()
one_sentence = True # or 100형태소


##### main #####################################################################################################################


if __name__ == '__main__':
    print('문장 생성봇!')

    while True:
        print('-' * 50)
        start_words = input('시작 단어: ')
        if start_words in ['/b', '/ㅠ']: break
    
        # if is_English_exist(start_words):
            # print('영어는 인식 못해 ㅜㅜ')
            # continue
        
        text = generate_words(start_words, model, kkma, morp_to_id, id_to_morp, one_sentence=one_sentence, verbose=False)
        if text is not None: print('\n'+text)
        model.reset_state()
