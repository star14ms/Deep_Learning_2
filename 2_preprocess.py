from modules.preprocess import preprocess, kkma
from modules.make_sentence import pos_to_sentence
from common.util import time
import time as t
import pickle
from os.path import isfile
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='data/YT_cmts_211101.txt', type=str, required=False,
                    help="path of youtube comments .txt file you can got after running 1_scrap_youtube_comment.py")
parser.add_argument("--save_path", default='saved_pkls/YT_cmts_211101_lang_corpus', type=str, required=False,
                    help="to save model directory")
parser.add_argument("--load_path", default=None, type=str, required=False,
                    help="to save model directory")
args = parser.parse_args()

##### 변수 선언 #########################################################################################################

start_time = t.time()

# 전처리할 텍스트 파일, 저장하고 불러올 파일
data_path = args.data_path
load_path = args.load_path
save_path = args.save_path

##### main #####################################################################################################################

if __name__ == '__main__':

    if isfile(f'{load_path}.pkl'):
        print('\n0) 기존 데이터에 추가하기...')
        with open(f'{load_path}.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        print(f'0) 데이터 불러오기 완료! ({load_path}.pkl)\n{time.str_delta(start_time)}')
    else: loaded_data = None


    print('\n1) 말뭉치 데이터 읽는 중...')
    with open(data_path, 'r', encoding='utf-8') as f:
        sentences = [line.lower() for line in f.read().splitlines()]
    # sentences = sentences[10:20] # ['이 문장은 시험용입니다.'] 
    print(f'1) 말뭉치 데이터 읽기 완료! (문장 개수: {len(sentences)}개)\n{time.str_delta(start_time)}')


    print('\n2) 형태소로 분해 중...')
    (vocab, corpus, sentences_ids) = preprocess(sentences, loaded_data, start_time=start_time)
    print(f'2) 형태소로 분해 완료! (형태소 개수: {vocab.n_morps}개)\n{time.str_delta(start_time)}')


    with open(f'{save_path}.pkl', 'wb') as f:
        pickle.dump({'vocab': vocab, 'corpus': corpus, 'sentences_ids': sentences_ids}, f)  
    print('\n저장한 형태소 사전 단어 수:', vocab.n_morps)

    # 테스트
    input("\n전처리 데이터 테스트 (Press Enter)")
    with open(f'{save_path}.pkl', 'rb') as f:
        (vocab, corpus, sentences_ids) = pickle.load(f).values()

    print(corpus[:100])
    for i in range(20):
        print(i, vocab.id2morp[i])
    
    while input() not in ['break','exit','/b','/e']: 
    # for i, sent_ids in enumerate(sentences_ids):
        # if input() in ['break','exit','/b','/e']: break
        print("-"*30)
        i = random.choice(range(len(sentences)))
        pos = kkma.pos(sentences[i])
        try: 
            decoded_sentence = pos_to_sentence(pos, verbose=True)
            print("\n문장 되돌리기:\n"+decoded_sentence)
        except: pass
        print("\n원본 문장:\n"+sentences[i])

