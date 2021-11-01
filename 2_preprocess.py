from modules.preprocess import preprocess, kkma
from modules.make_sentence import pos_to_sentence
from modules.translate_wclass import pos_ko
from common.util import time
import time as t
import pickle
from os.path import isfile
import random

##### 함수 선언 ###############################################################################################################

# 문장 데이터 읽어옴
def read_data(filepath, only_sentences_file=False):
    with open(filepath, 'r', encoding='utf-8') as f:
        if only_sentences_file: 
            return [line.lower() for line in f.read().splitlines()]
        
        return [line.lower() if line!="" and line[:2]!='$;' else '' for line in f.read().splitlines()]


##### 변수 선언 #########################################################################################################

start_time = t.time()

# 전처리할 텍스트 파일 정보
data_path = 'data/YT_cmts_211031.txt'
only_sentences_file = False

# 저장할 pkl 이름
save_pkl_name = 'saved_pkls/YT_cmts_211031_lang_corpus'

##### main #####################################################################################################################

if __name__ == '__main__':

    # if isfile(f'{save_pkl_name}.pkl'):
    #     print('\n0) 기존 데이터에 추가하기...')
    #     with open(f'{save_pkl_name}.pkl', 'rb') as f:
    #         loaded_data = pickle.load(f)
    #     print(f'0) 데이터 불러오기 완료! ({save_pkl_name}.pkl)\n{time.str_delta(start_time)}')
    # else: loaded_data = None


    print('\n1) 말뭉치 데이터 읽는 중...')
    sentences = read_data(data_path, only_sentences_file)
    # sentences = sentences[10:20] # ['이 문장은 시험용입니다.'] 
    print(f'1) 말뭉치 데이터 읽기 완료! (문장 개수: {len(sentences)}개)\n{time.str_delta(start_time)}')


    # print('\n2) 형태소로 분해 중...')
    # (lang, corpus, sentences_ids) = preprocess(sentences, loaded_data, start_time=start_time)
    # print(f'2) 형태소로 분해 완료! (형태소 개수: {lang.n_morps}개)\n{time.str_delta(start_time)}')


    # with open(f'{save_pkl_name}.pkl', 'wb') as f:
    #     pickle.dump({'lang': lang, 'corpus': corpus, 'sentences_ids': sentences_ids}, f)  
    # print('\n저장한 형태소 사전 단어 수:', lang.n_morps)


    with open(f'{save_pkl_name}.pkl', 'rb') as f:
        lang_data = pickle.load(f)
    lang, corpus, sentences_ids = lang_data.values()

    print(corpus[:100])
    for i in range(100):
        print(i, lang.id2morp[i])
    
    while input() not in ['break','exit','/b','/e']: 
    # for i, sent_ids in enumerate(sentences_ids):
        # if input() in ['break','exit','/b','/e']: break
        i = random.choice(range(len(sentences)))
        pos = kkma.pos(sentences[i])
        print(pos)
        print(pos_ko(pos))
        try: print(pos_to_sentence(pos, verbose=True))
        except: pass
        print(sentences[i].replace('\\','/'))

