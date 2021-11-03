# coding: utf-8
from common.np import *
from modules.rnnlm_gen import BetterRnnlmGen
import pickle
from konlpy.tag import Kkma
from modules.make_sentence import generate_sentence
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", default='ln_3680000 lt_5h59m43s ppl_266.3 BetterRnnlm params', type=str, required=False,
                    help="path of model (.pkl)")
parser.add_argument("--wordvec_size", default=512, type=int, required=False,
                    help="wordvec_size")
parser.add_argument("--hidden_size", default=512, type=int, required=False,
                    help="hidden_size")
parser.add_argument("--data_file", default="saved_pkls/YT_cmts_211101_lang_corpus.pkl", type=str, required=False,
                    help="path of .pkl file you got after running 2_preprocess.py")
parser.add_argument("--one_sentence", default=True, type=bool, required=False,
                    help="generate one sentence or 100형태소")
args = parser.parse_args()

##### 변수 선언 #########################################################################################################

with open(args.data_file, 'rb') as f:
    (lang, _, _) = pickle.load(f).values()
vocab_size = len(lang.id2morp)

model = BetterRnnlmGen(vocab_size, args.wordvec_size, args.hidden_size)
model.load_params(args.load_model)
kkma = Kkma()

##### main #####################################################################################################################


if __name__ == '__main__':
    print('문장 생성봇!')

    while True:
        print('-' * 50)
        start_words = input('시작 단어: ')
        if start_words in ['/b','/ㅠ','break','exit']: break
    
        # if is_English_exist(start_words):
            # print('영어는 인식 못해 ㅜㅜ')
            # continue
        
        text = generate_sentence(start_words, model, lang.morp2id, lang.id2morp, args.one_sentence, verbose=False)
        if text is not None: print('\n'+text)
        model.reset_state()
