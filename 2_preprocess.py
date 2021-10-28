from common.util import time
from common.preprocess import preprocess
import pickle
import time as t


##### 함수 정의 ###############################################################################################################


# 문장 데이터 읽어옴
def read_data(filepath, only_sentences_file=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        if only_sentences_file: return [line.lower() for line in f.read().splitlines()]

        data = [line.lower().split('\t') for line in f.read().splitlines()]
        data = data[1:] # header 제거
    
    return data


def save_morp_to_id(morps_to_id_pkl, corpus, morp_to_id, id_to_morp):
    with open(f'{morps_to_id_pkl}.pkl', 'ab') as f:
        pickle.dump((corpus, morp_to_id, id_to_morp), f)       


##### 변수 선언 #########################################################################################################


start_time = t.time()

# 전처리할 텍스트 파일 정보
dataset_path = 'dataset/YT_cmts.txt' # youtube_comments ratings
only_sentences_file = True

# 저장할 pkl 이름
sentences_pkl = 'saved_pkls/YT_cmts_morps_sentences_Kkma'
morps_to_id_pkl = 'saved_pkls/YT_cmts_morps_to_id_Kkma'


##### main #####################################################################################################################


if __name__ == '__main__':

    print('\n1) 말뭉치 데이터 읽는 중...')
    data = read_data(dataset_path, only_sentences_file=only_sentences_file)
    print('1) 말뭉치 데이터 읽기 완료! (문장 개수: {}개)'.format(len(data))+ f"\n{time.str_delta(start_time)}")
    

    print('\n2) 형태소로 분해 중...')
    sentences = [sentence_data[1] for sentence_data in data] if not only_sentences_file else data
    (lang, corpus) = preprocess(sentences, start_time=start_time, sentences_pkl=sentences_pkl)
    morp_to_id, id_to_morp = lang.word2index, lang.index2word
    print('2) 형태소로 분해 완료! (형태소 종류 수: {})'.format(len(id_to_morp))+ f"\n{time.str_delta(start_time)}")


    save_morp_to_id(morps_to_id_pkl, corpus, morp_to_id, id_to_morp)
    print('\n저장한 사전 길이:', len(id_to_morp))


    print(corpus[:100])
    for i in range(100):
        print(i, id_to_morp[i])