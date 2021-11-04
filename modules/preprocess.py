import pickle
import numpy as np
from torch.autograd.grad_mode import F
from tqdm import tqdm
import re
import sys
import konlpy
from konlpy.tag import Kkma, Okt

from common.util import time
from modules.translate_wclass import pos_ko, tag2morp
from modules.make_sentence import pos_to_sentence

MAX_LENGTH = 50

konlpy.jvm.init_jvm(jvmpath=None, max_heap_size=8192)
kkma = Kkma()
okt = Okt()


CHOSUNG_LIST = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 
    'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 
    'ㅌ', 'ㅍ', 'ㅎ'
]

JUNGSUNG_LIST = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ',
    'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ',
    'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]

JONGSUNG_LIST = [
    'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', # ' ', 
    'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',  
    'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
    'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ' 
]


class Vocab:
    """
    단어 사전
    ----------
    `self.name`: 사전 이름 (`str`)
    `self.morp2id`: 형태소를 id로 변환 (`dict`)
    `self.morp2count`: 각 형태소가 말뭉치에서 몇 번 나왔는지 (`dict`)
    `self.id2morp`: id를 형태소로 변환 (`dict`)
    `self.n_morps`: 사전에 들어있는 형태소 총 개수 (`int`)

    __init__ Parameters
    ----------
    `name`: 단어 사전 이름 (`str`)

    `morps`: 형태소 말뭉치 (konlpy kkma.pos() 형식) (`list[tuple(str, str)]`)

    Example
    ----------
    self.morp2id[morp][word_class] = id

    self.morp2count[morp][word_class] = count

    self.id2morp[id] = 형태소

    Functions
    ----------
    `addSentence()`: 문장 형태소 분석하여 형태소 사전에 추가
    `addWords()`: 형태소 말뭉치 사전에 추가
    """
    def __init__(self, name, morps):
        self.name = name
        self.morp2id = {}
        self.morp2count = {}
        self.id2morp = {0: "SOS", 1: "EOS"}
        self.n_morps = 2  # SOS 와 EOS 포함
        self.addWords(morps)

    def addSentence(self, sentence, Okt_nomalize, del_short_sound, del_repeated_latter):
        morps_sentences = make_morps_sentences([sentence], Okt_nomalize, del_short_sound, del_repeated_latter)
        
        morps = []
        for sentence in morps_sentences:
            for morp_wclass_tuple in sentence:
                morps.append(morp_wclass_tuple)
        morps = pos_ko(morps)

        self.addWords(morps)

    def addWords(self, morps):
        for morp, wclass in tqdm(morps): # morpheme: 형태소
            if morp not in self.morp2id:
                self.morp2id[morp] = {}
                self.morp2count[morp] = {}
            if wclass not in self.morp2id[morp]:
                self.morp2id[morp][wclass] = self.n_morps
                self.id2morp[self.n_morps] = morp
                self.morp2count[morp][wclass] = 1
                self.n_morps += 1
            else:
                self.morp2count[morp][wclass] += 1


def preprocess(texts, loaded_data=None, language='Korean', start_time=None, 
    Okt_nomalize=True, del_short_sound=True, del_repeated_latter=True
) -> (tuple):
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
    return `Vocab(Class), Corpus(np.ndarray[n]), Sentences_ids(list)`

    `Vocab`: 단어 사전

    `Corpus`: 말뭉치를 형태소 분해한 id뭉치

    `Sentences_ids`: 문장의 형태소 id 리스트 list[sentence1_ids, sentence2_ids, ...]
    """

    language = language.lower()
    if loaded_data:
        lang, loaded_corpus, loaded_sentence_ids = loaded_data.values()

    if language == 'korean':
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
            lang = Vocab("Korean", morps)
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


def make_morps_sentences(texts, Okt_nomalize=True, del_short_sound=True, del_repeated_latter=True) -> (list):
    """
    형태소 분석된 문장 list 만들기

    Parameters
    ----------
    `texts`: 전처리할 문장들 (`list[str]`)
    `Okt_nomalize`: konlpy okt.nomalize() 사용 여부 (`bool`)
    `del_short_sound`: 미완성 글자 지우기 여부 (`bool`)
    `del_repeated_latter`: 반복 글자 지우기 여부 (`bool`)

    Return
    ----------
    `list[kkma.pos(sent1), kkma.pos(sent2), ...]`

    `list[list[tuple(str, str)]]`
    """
    len_text = len(texts)
    morps_sentences = []
    n_splited = 0

    emoji_pattern = re.compile("["
        # u"\U0001F600-\U0001F64F"  # emoticons
        # u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        # u"\U0001F680-\U0001F6FF"  # transport & map symbols
        # u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        # u"\U00002702-\U000027B0"
        # u"\U000024C2-\U0001F251"
        # u"\U00002500-\U00002BEF"  # chinese char
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        # u"\u2640-\u2642"
        # u"\u2600-\u2B55"
        # u"\u200d"
        # u"\u23cf"
        # u"\u23e9"
        # u"\u231a"
        # u"\ufe0f"  # dingbats
        # u"\u3030"
                           "]+", flags=re.UNICODE)

    for idx, sentence in enumerate(texts):
        sys.stdout.write(('\r%d / %d | %s' % (idx+1, len_text, (
            sentence[:25]+'...' if len(sentence)>=25 else sentence).ljust(32))))
        sys.stdout.flush()
        
        # 0. print() 못하는 이모티콘 삭제
        sentence2 = emoji_pattern.sub(r'', sentence) # no emoji
        # if sentence != sentence2: print('\n이모티콘 삭제->', sentence2)
        
        # 0. 문자가 하나도 없으면 스킵
        if sentence2 == '': continue

        only_spaces = True
        for char in sentence2: 
            if char!=' ':
                only_spaces = False 
                break
        if only_spaces: continue
        
        # 1. okt 형태소 분석기의 normalize 전처리 사용
        normalized = okt.normalize(sentence2) if Okt_nomalize else sentence2
        # if sentence != normalized:
            # try: print('\nOkt 정규화\n-> {}\n-> {}\n'.format(sentence, normalized))
            # except: pass

        # 2. 초성, 중성, 종성만 19자 이상 반복시 줄이기 (형태소 분석 지연 방지)
        normalized2 = nomalize_short_sounds(normalized) if del_short_sound else normalized
        if normalized2 == False: continue
        # if normalized != normalized2: print('\n반복 초성 삭제\n-> {}\n-> {}\n'.format(normalized, normalized2))
        
        # 3. 반복되는 문자 간략화
        normalized3 = delete_repeated_latter(normalized2) if del_repeated_latter else normalized2 ### sentence
        # if normalized2 != normalized3: print('\n반복 글자 삭제\n-> {}\n-> {}\n'.format(normalized2, normalized3))
        
        # 4. 형태소 분해 후 문장 단위로 쪼개서 저장
        pos = kkma.pos(normalized3)
        splited_pos = split_pos_into_sentence(pos)
        if len(splited_pos) > 1: n_splited += 1

        for pos in splited_pos:
            morps_sentences.append(pos)

    return morps_sentences


def nomalize_short_sounds(sentence: str, skip_num: int = 30, max_n_short: int = 18) -> (str):
    """
    반복되는 초성, 중성, 종성 제거 (미완성된 글자)

    (`kkma 형태소 분석 때 무한 지연 방지를 위한 함수`) 

    Parameters
    ----------
    `skip_num`: 미완성 글자가 총합 n개를 넘으면 return False
    `max_n_short`: 미완성 글자를 연속 n개까지만 남기고 나머지는 삭제
    """
    short_sounds = CHOSUNG_LIST + JUNGSUNG_LIST + JONGSUNG_LIST
    n_short_sound = 0
    continuous_short_sound = 0
    sentence2 = ""
    for latter in sentence:
        if latter in short_sounds:
            continuous_short_sound += 1
            n_short_sound += 1
        else:
            continuous_short_sound = 0

        if continuous_short_sound <= max_n_short:
            sentence2 = sentence2 + latter

    if n_short_sound > skip_num: return False

    return sentence2


def delete_repeated_latter(sentence: str, len_kkma_delay: int = 18) -> (str):
    """
    Parameters
    ----------
    `skip_num`: 미완성 글자가 총합 n개를 넘으면 return False
    `len_kkma_delay`: konlpy kkma 형태소 분석을 지연시키는 연속된 글자 최소 길이
    """
    repeator_size = 0
    while True:
        target = sentence[:repeator_size+1]
        if target[-1] == ' ': return sentence # 한칸 뛰었기 때문에 더 이상 반복 키워드로 취급하지 않음
        index = -1
        idxs = []
        while True:
            last_index = index
            index = sentence.find(target, index+1)
            if index == -1 or \
                (len(idxs)>=2 and idxs[-1]-idxs[-2] != index-idxs[-1]):
                break
            idxs.append(index)

        if len(idxs) >= 4 and (
            idxs[3]-idxs[2] == idxs[2]-idxs[1] == idxs[1]-idxs[0]):
            repeator_size += 1
            if idxs[1]-idxs[0] == repeator_size:
                break
        else:
            return sentence

    if 1 < repeator_size:
        # print('\n반복 문자:', target, idxs)
        return target*(4 if repeator_size<=4 else len_kkma_delay//repeator_size) + (
            sentence[last_index+repeator_size:] if last_index + repeator_size < len(sentence) else "")
    else:
        return sentence


def split_pos_into_sentence(pos, MAX_LENGTH=MAX_LENGTH) -> (list):
    """
    konlpy kkma.pos() 로 형태소 분석하여 나온 list를 문장별로 나누기
    
    Parameters
    ----------
    `pos`: konlpy kkma.pos() return 값 (list[str, str])
    `MAX_LENGTH`: 형태소 개수 MAX_LENGTH 이내에서 문장 합쳐서 한 문장으로 취급
    """
    if len(pos) > MAX_LENGTH:
        
        # 문장들 길이 구하기
        lens_sent = []
        idx_former_end = -1
        end_sentence = False
        for idx, (morp, wclass) in enumerate(pos):
            if 'EF' in wclass: end_sentence = True
            
            if morp=='.' and wclass=='SF' or idx==len(pos)-1 or \
                (end_sentence and \
                    wclass in ['EFN','EFQ','EFO','EFA','EFI','EFR','EMO','SW'] and \
                    pos[idx+1][1] not in ['EMO','SW']):
                end_sentence = False
                len_sent = idx - idx_former_end
                lens_sent.append(len_sent)
                idx_former_end = idx

        # MAX_LENGTH보다 작게 유지하며 길이 합치기
        for idx, len_sent in enumerate(lens_sent):
            if idx==0: continue
            else: 
                len_merged = len_sent + lens_sent[idx-1]
                if len_merged <= MAX_LENGTH:
                    lens_sent[idx] = len_merged
                    lens_sent[idx-1] = 0

        # 합쳐진 길이들로 문장 자르기
        # print()
        # print(lens_sent)
        # print(pos)
        splited_morps_sent = []
        idx = 0
        for len_sent in lens_sent:
            if len_sent==0 or len_sent > MAX_LENGTH: continue
            splited_morps_sent.append(pos[idx:idx+len_sent])
            idx += len_sent
        
        # for morps_sent in splited_morps_sent:
        #     print("\n", pos_to_sentence(morps_sent))
        # print()
        return splited_morps_sent
    else:
        return [pos]


