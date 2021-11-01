from modules.translate_wclass import pos_ko
import numpy as np

from modules.unicode import join_jamos
from jamo import h2j, j2hcj

import konlpy
from konlpy.tag import Kkma, Okt
konlpy.jvm.init_jvm(jvmpath=None, max_heap_size=8192)
kkma = Kkma()


shortening_words = [ 
    ('가아','가'),('나아','나'),('다아','다'),('라아','라'),('마아','마'),
    ('바아','바'),('사아','사'),('아아','아'),('자아','자'),('차아','차'),
    ('카아','카'),('타아','타'),('파아','파'),('하아','하'),('까아','까'),

    ('거어','거'),('너어','너'),('더어','더'),('러어','러'),('머어','머'),
    ('버어','버'),('서어','서'),('어어','어'),('저어','저'),('처어','처'),
    ('커어','커'),('터어','터'),('퍼어','퍼'),('허어','허'),

    ('하어','해'),('하었','했'),('쓰었','썼'),('되어','돼'),('나었','났'),
    ('나는','난'),('너는','넌'),('오아서','와서'),('스어','서'),('쓰어','써'),
    ('하이라','해라'),('보았','봤'),('그리하여도','그래도'),('때었','땠'),
    ('주어야','줘야'),

    ('시었','셨'),('끼었','꼈'),('지었','졌'),
    ('이었','였'),('히었','혔'),('리었','렸'),('기었','겼'),

    ('알니','아니'),('걸ㄴ','건'),('닿ㄹ라','달라'),('알ㄴ','안'),('들ㄴ','든'),
    ('몰ㄹ','몰'),('낳ㄴ','난'),('밉ㄴ','미운'),('어렵ㄴ','어려운'),
    (' n ',' n'),('알ㄹ거','알 거'),('어떻ㄴ거','어떤거'),('느어','너'),
    ('고맙어','고마워'),('고맙었','고마웠')
] # Konlpy kkma 기준


def is_English_exist(string):
    for latter in 'abcdefghijklmnopqrstuvwxyz':
        if string.count(latter) != 0:
            return True

    return False


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key


def is_space_needed(key, pre_key, in_quotation_mark):
    space = True

    for wclass in ['조사', '어미', '지정사', '마침표', '쉼표', '줄임표', '의존 명사', '붙임표', '접미사', '따옴표','기타기호']:
        if wclass in key:
            space = False
            break
    
    if ((pre_key in ['숫자', '수사', '붙임표(물결,숨김,빠짐)']) or
        # (key=='숫자' and pre_key=='기타기호(논리수학,화폐)') or 
        # (key=='기타기호(논리수학,화폐)' and pre_key=='숫자') or 
        (pre_key=='따옴표,괄호표,줄표') or 
        # (key=='외국어' and pre_key=='외국어'):
        (key=='한자' and pre_key=='한자')): 
        space = False
 
    if not in_quotation_mark and '따옴표,괄호표,줄표' in [key, pre_key]:
        space = True

    return space


def ids_to_sentence(word_ids, morp2id, id2morp, verbose=False):
    text = ''
    pre_key = None
    in_quotation_mark = False

    for id in word_ids:
        morp = id2morp[id]
        key = get_key(morp2id[morp], id)
        space = is_space_needed(key, pre_key, in_quotation_mark)
        if verbose: print(morp.ljust(10), str(space).ljust(5), key)

        if key=='따옴표,괄호표,줄표':
            in_quotation_mark = False if in_quotation_mark else True
        pre_key = key
        text = text + (' ' if space and text!='' else '') + morp

    # print('\n'+text) # '\n'+sentence
    jamo = j2hcj(h2j(text))
    sentence = join_jamos(jamo)

    for shortening in shortening_words:
        sentence = sentence.replace(*shortening)
    return sentence


def pos_to_sentence(pos, verbose=False):
    text = ''
    pre_wclass = None
    in_quotation_mark = False

    for (morp, wclass) in pos_ko(pos):
        space = is_space_needed(wclass, pre_wclass, in_quotation_mark)
        if verbose: print(morp.ljust(10), str(space).ljust(5), wclass)

        if wclass=='따옴표,괄호표,줄표':
            in_quotation_mark = False if in_quotation_mark else True
        pre_wclass = wclass
        text = text + (' ' if space and text!='' else '') + morp
    jamo = j2hcj(h2j(text))
    sentence = join_jamos(jamo)

    for shortening in shortening_words:
        sentence = sentence.replace(*shortening)
    return sentence


def generate_sentence(start_words, model, morp2id, id2morp, one_sentence=True, verbose=False):
    pos = pos_ko(kkma.pos(start_words))

    try:
        start_ids = [morp2id[morp][wclass] for morp, wclass in pos]
    except KeyError:
        print('그 말은 단어 사전에 아직 없어 ㅜㅜ')
        return
        
    if len(start_ids) == 1:
        word_ids = model.generate(
            start_ids[-1], one_sentence=one_sentence, id2morp=id2morp)
    else:
        for x in start_ids[:-1]:
            x = np.array(x).reshape(1, 1)
            model.predict(x)
        word_ids = start_ids[:-1] + model.generate(
            start_ids[-1], one_sentence=one_sentence, id2morp=id2morp)

    return ids_to_sentence(word_ids, morp2id, id2morp, verbose=verbose)
