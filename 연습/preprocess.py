from common.util import preprocess, time
import pickle
import time as t

start_time = t.time()
dataset_path = 'dataset/YT_cmts.txt' # youtube_comments ratings
only_sentences_file = True
load_morp_to_id_pkl = None #'saved_pkls/YT_cmts_morps_to_id_Kkma'

# 저장할 pkl 이름
sentences_pkl = 'saved_pkls/YT_cmts_morps_sentences_Kkma'
morps_to_id_pkl = 'saved_pkls/YT_cmts_morps_to_id_Kkma'
if load_morp_to_id_pkl == morps_to_id_pkl:
    print('불러올 이름과 저장할 이름이 같음 (morp_to_id)')
    exit()
    
# 문장 데이터 읽어옴
def read_data(filepath, only_sentences_file=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        if only_sentences_file: return [line.lower() for line in f.read().splitlines()]

        data = [line.lower().split('\t') for line in f.read().splitlines()]
        data = data[1:] # header 제거
    
    return data
        
# 리뷰 파일 읽어오기
print('\n1) 말뭉치 데이터 읽는 중...')
data = read_data(dataset_path, only_sentences_file=only_sentences_file)
print('1) 말뭉치 데이터 읽기 완료! (문장 개수: {}개)'.format(len(data))+ f"\n{time.str_hms_delta(start_time)}")

# 문장만 가져와, 각 문장을 형태소로 분해하면서 각 형태소 id 만들기
if load_morp_to_id_pkl != None:
    with open(f'{load_morp_to_id_pkl}.pkl', 'rb') as f:
        (_, morp_to_id, id_to_morp) = pickle.load(f)
    print('불러온 사전 길이:', len(id_to_morp))

print('\n2) 형태소로 분해 중...')
sentences = [sentence_data[1] for sentence_data in data] if not only_sentences_file else data
(corpus, morp_to_id, id_to_morp) = preprocess(sentences, start_time=start_time, sentences_pkl=sentences_pkl)
print('2) 형태소로 분해 완료! (형태소 종류 수: {})'.format(len(id_to_morp))+ f"\n{time.str_hms_delta(start_time)}")

with open(f'{morps_to_id_pkl}.pkl', 'ab') as f:
    pickle.dump((corpus, morp_to_id, id_to_morp), f)

print('\n저장한 사전 길이:', len(id_to_morp))
print(corpus[:100])
for i in range(100):
    print(i, id_to_morp[i])