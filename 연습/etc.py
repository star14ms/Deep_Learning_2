# 형태소 분해 테스트
from konlpy.tag import Okt, Kkma
from modules.translate_wclass import translate_wclass as ts_wclass
okt = Okt()
kkma = Kkma()

def test_pos(sentence):
    print(sentence)
    b = okt.normalize(sentence)
    print(b)
    c = kkma.pos(b)
    for pos in ts_wclass(c): print(pos)

# test_pos('개들의 전쟁2 나오나요? 나오면 1빠로 보고 싶음')


# 형태소 분해된 문장 다시 문장으로 만들기
from common.util import word_ids_to_sentence
from tqdm import tqdm
import pickle

with open('saved_pkls/YT_cmts_morps_to_id_Kkma.pkl', 'rb') as f:
    (corpus, morp_to_id, id_to_morp) = pickle.load(f)

print('형태소 총 갯수 %d, 종류 수 %d'% (len(corpus), len(id_to_morp)))
sentence = word_ids_to_sentence(corpus[-100:], morp_to_id, id_to_morp)
print(sentence)


# # 특정 종류의 형태소 찾기, 특정 형태소의 종류 찾기
# morps = []
# for morp in tqdm(morp_to_id):
#     for wclass in morp_to_id[morp].keys():
#         if wclass in ['명사추정범주']:
#             morps.append(morp)
#             break
#     # if morp == '땃쥐보고싶어':
#     #     print(morp_to_id[morp])
# print(morps, len(morps))


def filter_or_merge_already_saved_comments(pre_txt, target_txt, new_txt, merge=False):
    with open(f'{pre_txt}.txt', 'r', encoding="utf-8") as f:
        cmts1 = [sentence for sentence in f.read().splitlines()]
    
    with open(f'{target_txt}.txt', 'r', encoding="utf-8") as f:
        cmts2 = [sentence for sentence in f.read().splitlines()]

    if merge:
        with open(f'{new_txt}.txt', 'a', encoding="utf-8") as f:
            for cmt in tqdm(cmts1+cmts2):
                f.write(cmt + ('.\n' if cmt[-1] not in '.?!' else '\n'))
    else:
        with open(f'{new_txt}.txt', 'a', encoding="utf-8") as f:
            for cmt2 in tqdm(cmts2):
                if cmt2 not in cmts1:
                    f.write(cmt2+'\n')

# pre_txt = 'dataset/YT_cmts1'
# target_txt = 'dataset/YT_cmts2'
# new_txt = 'dataset/YT_cmts'
# filter_or_merge_already_saved_comments(pre_txt, target_txt, new_txt, merge=True)
# exit()


# 문장 데이터 읽어옴
def read_data(filepath, only_sentences_file=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        if only_sentences_file: 
            return [line.lower() for line in f.read().splitlines()]
        
        return [line.lower() if line!="" and line[:2]!='$;' else '' for line in f.read().splitlines()]


def preprocess(texts, loaded_data=None, language='Korean', splited_sentence=True, 
    start_time=None, Okt_nomalize=True, del_short_sound=True, del_repeated_latter=True
) -> tuple:
    """
    데이터 전처리
    ----------

    Parameters
    ----------
    `texts`: 전처리할 문장들 (`list[str]`)
    `loaded_data`: 전처리된 데이터 파일 로드한 것 (`dict`)
    `start_time`: 시작 시간 (`float` `time.time()`)
    `Okt_nomalize`: konlpy okt.nomalize() 사용 여부 (`bool`)
    `del_short_sound`: 미완성 글자 지우기 여부 (`bool`)
    `del_repeated_latter`: 반복 글자 지우기 여부 (`bool`)

    Return
    ----------
    return `Lang(Class), Corpus(np.ndarray[n]), Sentences_ids(list)`

    `Lang`: vocab

    `Corpus`: 말뭉치를 형태소 분해한 id뭉치

    `Sentences_ids`: list[sentence1_ids, sentence2_ids, ...]
    """

    language = language.lower()
    if loaded_data:
        lang, loaded_corpus, loaded_sentence_ids = loaded_data.values()

    if language == 'english':
        if splited_sentence: 
            texts = ' '.join(texts)
        texts = texts.lower()
        texts = texts.replace('.', ' .')
        words = texts.split(' ')

        word_to_id = {}
        id_to_word = {}
        
        for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word
        
        corpus = np.array([word_to_id[w] for w in words])

        return corpus, word_to_id, id_to_word
        
    elif language == 'korean':
        morps_sentences = make_morps_sentences(texts, Okt_nomalize, del_short_sound, del_repeated_latter)
        if start_time != None: print(time.str_delta(start_time), "형태소 분해 완료!")

        morps = []
        for sentence in morps_sentences:
            for morp_wclass_tuple in sentence:
                morps.append(morp_wclass_tuple)
        morps = pos_ko(morps)
        
        if loaded_data:
            lang.addWords(morps)
        else:
            lang = Lang("Korean", morps)
        if start_time != None: print(time.str_delta(start_time), "형태소 사전 만들기 완료!")
        
        corpus = np.array([lang.morp2id[morp][wclass] for morp, wclass in tqdm(morps)])
        if start_time != None: print(time.str_delta(start_time), "데이터를 형태소 id로 표현 완료!")
        
        sentences_ids = []
        for sentence in morps_sentences:
            sentence_ids = [lang.morp2id[morp][tag2morp(wclass)] for morp, wclass in sentence]
            sentences_ids.append(sentence_ids)
        
        if loaded_data:
            return (lang, np.hstack((loaded_corpus, corpus)), loaded_sentence_ids+sentences_ids)
        else:
            return (lang, corpus, sentences_ids)