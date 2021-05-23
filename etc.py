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