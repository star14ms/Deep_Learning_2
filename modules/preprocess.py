import pickle
import numpy as np
from tqdm import tqdm
import re
import sys
import konlpy
from konlpy.tag import Kkma, Okt

from common.util import time
from modules.translate_wclass import pos_ko, tag2morp


konlpy.jvm.init_jvm(jvmpath=None, max_heap_size=8192)
kkma = Kkma()
okt = Okt()


class Lang:
    def __init__(self, name, morps):
        self.name = name
        self.morp2id = {}
        self.morp2count = {}
        self.id2morp = {0: "SOS", 1: "EOS"}
        self.n_morps = 2  # SOS 와 EOS 포함
        self.addWords(morps)

    def addSentence(self, sentence, Okt_nomalize, cho_sung_nomalize, del_repeat_latter):
        morps_sentences = make_morps_sentences([sentence], Okt_nomalize, cho_sung_nomalize, del_repeat_latter)
        
        morps = []
        for sentence in morps_sentences:
            for morp_wclass_tuple in sentence:
                morps.append(morp_wclass_tuple)
        morps = pos_ko(morps, translate_level=3)

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


def preprocess(texts, language='Korean', splited_sentence=True, 
    start_time=None, sentences_pkl=None,
    Okt_nomalize=True, cho_sung_nomalize=True, del_repeat_latter=True):

    language = language.lower()

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
        # morps_sentences = [Kkma().pos(Okt().normalize(sentence)) for sentence in tqdm(texts)] # morpheme: 형태소
        morps_sentences = make_morps_sentences(texts, Okt_nomalize, cho_sung_nomalize, del_repeat_latter)
        if start_time != None: print(time.str_delta(start_time), "형태소 분해 완료!")
        
        if sentences_pkl==None: sentences_pkl = 'morps_to_id_Kkma'
        with open(f'{sentences_pkl}.pkl', 'wb') as f:
            pickle.dump(morps_sentences, f)
        # with open(f'{sentences_pkl}.pkl', 'rb') as f:
            # morps_sentences = pickle.load(f)

        morps = []
        for sentence in morps_sentences:
            for morp_wclass_tuple in sentence:
                morps.append(morp_wclass_tuple)
        morps = pos_ko(morps)

        lang = Lang("Korean", morps)
        if start_time != None: print(time.str_delta(start_time), "형태소 사전 만들기 완료!")
        
        corpus = np.array([lang.morp2id[morp][wclass] for morp, wclass in tqdm(morps)])
        if start_time != None: print(time.str_delta(start_time), "데이터를 형태소 id로 표현 완료!")
        
        sentences_ids = []
        for sentence in morps_sentences:
            sentence_ids = [lang.morp2id[morp][tag2morp(wclass)] for morp, wclass in sentence]
            sentences_ids.append(sentence_ids)
        
        return (lang, corpus, sentences_ids)


def make_morps_sentences(texts, Okt_nomalize=True, cho_sung_nomalize=True, del_repeat_latter=True):
    len_text = len(texts)
    morps_sentences = []

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
        sys.stdout.write(('\r%d / %d | %s' % (idx, len_text, (
            sentence[:25]+'...' if len(sentence)>=25 else sentence).ljust(32))))
        sys.stdout.flush()
        
        sentence2 = emoji_pattern.sub(r'', sentence) # no emoji
        # if sentence != sentence2: print('\n이모티콘 삭제->', sentence2)
        
        if sentence2 == '': continue
        only_spaces = True
        for char in sentence2: only_spaces = False if char!=' ' else only_spaces
        if only_spaces: continue
        
        # okt 형태소 분석기의 normalize 전처리 사용
        normalized = okt.normalize(sentence2) if Okt_nomalize else sentence2
        # if sentence != normalized:
            # try:
            #     print('\nOkt 정규화\n-> {}\n-> {}\n'.format(sentence, normalized))
            # except:
            #     pass
            #     import json
            #     print('\nOkt 정규화\n-> {}\n-> {}\n'.format(json.loads(sentence), normalized))
        
        # 초성만 19자 이상 반복시 줄이기 (분석 지연 방지)
        normalized2 = nomalize_cho_sungs(normalized) if cho_sung_nomalize else normalized
        if normalized != normalized2: print('\n반복 초성 삭제\n-> {}\n-> {}\n'.format(normalized, normalized2))
        
        # 반복되는 문자 간략화
        normalized3 = delete_repeat_latter(normalized2) if del_repeat_latter else normalized2 ### sentence
        if normalized2 != normalized3: print('\n반복 글자 삭제\n-> {}\n-> {}\n'.format(normalized2, normalized3))
        
        morps_sentences.append(kkma.pos(normalized3))
        # if idx==0: print(morps_sentences[-1])
    return morps_sentences


def nomalize_cho_sungs(sentence):
    continuous_cho_sung = 0
    sentence2 = ""
    for latter in sentence:
        if latter in 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㄳㄶㄵㄺㄻㄼㄽㄾㅀㅄㆍ':
            continuous_cho_sung += 1
        else:
            continuous_cho_sung = 0

        if continuous_cho_sung <= 18:
            sentence2 = sentence2 + latter

    return sentence2


def delete_repeat_latter(sentence, kkma_error_occur_len=18):
    repeator_size = 0
    while True:
        target = sentence[:repeator_size+1]
        if target[-1] == ' ': return sentence # 한칸 뛰었기 때문에 더 이상 반복 키워드로 취급하지 않음
        index = -1
        idxs = []
        while True:
            last_index = index
            index = sentence.find(target, index+1)
            if index == -1:
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
        return target*(4 if repeator_size<=4 else kkma_error_occur_len//repeator_size) + (
            sentence[last_index+repeator_size:] if last_index + repeator_size < len(sentence) else "")
    else:
        return sentence
