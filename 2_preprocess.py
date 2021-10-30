from modules.preprocess import preprocess, kkma
from modules.make_sentence import ids_to_sentence
from modules.translate_wclass import pos_ko
from common.util import time
import time as t
import pickle
import random

##### 함수 정의 ###############################################################################################################

# 문장 데이터 읽어옴
def read_data(filepath, only_sentences_file=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        if only_sentences_file: return [line.lower() for line in f.read().splitlines()]

        data = [line.lower().split('\t') for line in f.read().splitlines()]
        data = data[1:] # header 제거
    
    return data

##### 변수 선언 #########################################################################################################

start_time = t.time()

# 전처리할 텍스트 파일 정보
data_path = 'data/YT_cmts_211029.txt' # youtube_comments ratings
only_sentences_file = True

# 저장할 pkl 이름
lang_corpus_pkl = 'saved_pkls/YT_cmts_lang_corpus'

##### main #####################################################################################################################

if __name__ == '__main__':

    print('\n1) 말뭉치 데이터 읽는 중...')
    data = read_data(data_path, only_sentences_file=only_sentences_file)
    # data = data[:100]
    sentences = [sentence[1] for sentence in data] if not only_sentences_file else data
    print(f'1) 말뭉치 데이터 읽기 완료! (문장 개수: {len(data)}개)\n{time.str_delta(start_time)}')
    

    print('\n2) 형태소로 분해 중...')
    (lang, corpus, sentences_ids) = preprocess(sentences, start_time=start_time)
    print(f'2) 형태소로 분해 완료! (형태소 개수: {lang.n_morps}개)\n{time.str_delta(start_time)}')


    with open(f'{lang_corpus_pkl}.pkl', 'wb') as f:
        pickle.dump({'lang': lang, 'corpus': corpus, 'sentences_ids': sentences_ids}, f)  
    print('\n저장한 형태소 사전 단어 수:', lang.n_morps)


    with open(f'{lang_corpus_pkl}.pkl', 'rb') as f:
        lang_data = pickle.load(f)
    lang, corpus, sentences_ids = lang_data.values()

    print(corpus[:100])
    for i in range(100):
        print(i, lang.id2morp[i])
    
    while input() not in ['break','exit','/b','/e']: 
    # for i, sent_ids in enumerate(sentences_ids):
        # if input() in ['break','exit','/b','/e']: break
        i = random.choice(range(len(data)))
        print(pos_ko(kkma.pos(data[i])))
        print(ids_to_sentence(sentences_ids[i], lang.morp2id, lang.id2morp, verbose=True))
        print(data[i])

