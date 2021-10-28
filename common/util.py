import numpy as np
from konlpy.tag import Kkma, Okt
import time as t

from numpy.lib.function_base import iterable
from modules.translate_wclass import translate_wclass as ts_wclass
import sys
import os
from tqdm import tqdm
 
import konlpy
konlpy.jvm.init_jvm(jvmpath=None, max_heap_size=8192)
kkma = Kkma()
okt = Okt()

################################################################################################################################

def create_co_matrix(corpus, vocab_size, window_size=1): # co-occurrence matrix
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print("%s(을)를 찾을 수 없습니다" % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0 
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print('%s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps) ### S[j]*S[i]
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
    return M


def create_contexts_target(corpus, window_size=1):
    '''コンテキストとターゲットの作成

    :param corpus: コーパス（単語IDのリスト）
    :param window_size: ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return:
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def eval_perplexity(model, corpus, batch_size=20, time_size=35):
    print('evaluating perplexity ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(model, question, correct, id_to_char,
                 verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 頭の区切り文字
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 文字列へ変換
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        print('---')

    return 1 if guess == correct else 0


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s is not found' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x


class time:
    def sec_to_hms(second):
            second = int(second)
            h, m, s = (second // 3600), (second//60 - second//3600*60), (second % 60)
            return h, m, s
    
    def str_delta(start_time, hms=False, rjust=False, join=':'):
        time_delta = t.time() - start_time
        h, m, s = time.sec_to_hms(time_delta)
        if not hms:
            return "{1}{0}{2:02d}{0}{3:02d}".format(join, h, m, s)
        elif rjust: 
            m, s = str(m).rjust(2), str(s).rjust(2)
        
        return str(f"{h}h{join}{m}m{join}{s}s")
        
    def str(second, hms=False, rjust=False, join=':'):
            h, m, s = time.sec_to_hms(second)
            if not hms:
                return "{1}{0}{2:02d}{0}{3:02d}".format(join, h, m, s)
            elif rjust: 
                m, s = str(m).rjust(2), str(s).rjust(2)
            
            return str(f"{h}h{join}{m}m{join}{s}s")


import logging
def __get_logger():
    """로거 인스턴스 반환
    """

    __logger = logging.getLogger('logger')

    # 로그 포멧 정의
    formatter = logging.Formatter(
        '\n"%(pathname)s", line %(lineno)d, in %(module)s\n%(levelname)-8s: %(message)s')
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger


def nouns(corpus, morp_to_id, id_to_morp):
    nouns = []
    for id in tqdm(corpus):
        if id_to_morp[id] in 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ':
            continue
        for wclass in morp_to_id[id_to_morp[id]].keys():
            if (wclass=='명사' and morp_to_id[id_to_morp[id]]['명사']==id or
                wclass=='명사추정범주' and morp_to_id[id_to_morp[id]]['명사추정범주']==id): # '명사', '고유명사'
                nouns.append(id_to_morp[id])
                break

    return nouns

################################################################################################################################

def is_English_exist(string):
    for latter in 'abcdefghijklmnopqrstuvwxyz':
        if string.count(latter) != 0:
            return True

    return False

def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key

from modules.unicode import join_jamos
from jamo import h2j, j2hcj
def word_ids_to_sentence(word_ids, morp_to_id, id_to_morp, verbose=False):
    text = ''
    pre_key = None
    for id in word_ids:
        key = get_key(morp_to_id[id_to_morp[id]], id)
        space = True
        for wclass in ['조사', '어미', '지정사', '마침표', '쉼표', '줄임표', '의존 명사', '붙임표', '접미사', '따옴표']:
            if wclass in key:
                space = False
                break
        
        if (key=='숫자' and pre_key=='기타기호(논리수학,화폐)' or 
            key=='외국어' and pre_key=='따옴표,괄호표,줄표' or 
            key=='한자' and pre_key=='한자'): #or
            # key=='외국어' and pre_key=='외국어'):
            space = False
        pre_key = key

        if verbose: print(id_to_morp[id].ljust(10), str(space).ljust(5), key)
        text = text + (' ' if space and text!='' else '') + id_to_morp[id]

    print('\n'+text) # '\n'+sentence
    jamo = j2hcj(h2j(text))
    sentence = join_jamos(jamo)
    return sentence

def generate_words(start_words, model, kkma, morp_to_id, id_to_morp, one_sentence=True, verbose=False):
    pos = ts_wclass(kkma.pos(start_words), kkma=True, translate_level=3)
    try:
        start_ids = [morp_to_id[morp][wclass] for morp, wclass in pos]
    except KeyError:
        print('그 말은 단어 사전에 아직 없어 ㅜㅜ')
        return
        
    if len(start_ids) == 1:
        word_ids = model.generate(
            start_ids[-1], one_sentence=one_sentence, id_to_morp=id_to_morp)
    else:
        for x in start_ids[:-1]:
            x = np.array(x).reshape(1, 1)
            model.predict(x)
        word_ids = start_ids[:-1] + model.generate(
            start_ids[-1], one_sentence=one_sentence, id_to_morp=id_to_morp)

    return word_ids_to_sentence(word_ids, morp_to_id, id_to_morp, verbose=verbose)

################################################################################################################################
