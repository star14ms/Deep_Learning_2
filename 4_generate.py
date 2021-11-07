# coding: utf-8
from model import LSTM
from common.np import *
from common.config import Config
from modules.rnnlm_gen import BetterRnnlmGen
from modules.make_sentence import generate_sentence
import pickle
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", default='saved_models/LSTM ep_10 ppl_213.5.pth', type=str, required=False,
                    help="path of model (.pkl, .pth) you can got after running 3_train_better_rnnlm.py or train.py")
parser.add_argument("--config", default="config_LSTM.json", type=str, required=False,
                    help="config file")
parser.add_argument("--data_file", default="saved_pkls/YT_cmts_211101~06_vocab_corpus.pkl", type=str, required=False,
                    help="path of .pkl file you can got after running 2_preprocess.py")
parser.add_argument("--one_sentence", default=True, type=bool, required=False,
                    help="generate one sentence or 100형태소")
parser.add_argument("--save_dir", default="saved_models", type=str, required=False,
                    help="to save model directory")
args = parser.parse_args()

##### 변수 선언 #########################################################################################################

with open(args.data_file, 'rb') as f:
    (vocab, _, _) = pickle.load(f).values()

config = Config.load(args.config)
config.vocab_size = vocab.n_morps

if "LSTM" in args.config:
    model = LSTM(config)
    model.load(args.load_model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
else:
    model = BetterRnnlmGen(config)
    model.load_params(args.load_model, args.save_dir)

end = '[EOS]'

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
        
        text = generate_sentence(start_words, model, vocab.morp2id, vocab.id2morp, args.one_sentence, verbose=False, end=end)
        if text is not None: print('\n'+text)

