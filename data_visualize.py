from konlpy.corpus import kolaw
from nltk import Text
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import pickle
from common.util import nouns
import numpy as np

# 학습 데이터 파일 경로
data_file = r'data/YT_cmts_211101.txt'

# with open('saved_pkls/YT_cmts_morps_to_id_Kkma.pkl', 'rb') as f:
#     (corpus, morp_to_id, id_to_morp) = pickle.load(f)
with open('saved_pkls/YT_cmts_211101_lang_corpus.pkl', 'rb') as f:
    (lang, corpus, _) = pickle.load(f).values()
morp_to_id, id_to_morp = lang.morp2id, lang.id2morp

# 폰트 설정
font_path = r'common/font/주아체.ttf' # 한글 폰트 경로
font_name = fm.FontProperties(fname=font_path, size=50).get_name()
print(font_name)
plt.rc('font', family=font_name)
# plt.rcParams["font.family"] = "배달의민족 주아" #? 자꾸 인식 못함
plt.rcParams["font.size"] = 13

# 학습 데이터의 단어들 비중 순서로 그래프 그리기
# c = open(data_file, encoding='utf8').read() # constitution, ratings
# words = Text(Okt().nouns(c[:10000]), name="kolaw")

print('\n명사만 추출 중...')
words = Text(nouns(corpus, morp_to_id, id_to_morp), name="morps")
samples = [item for item, _ in words.vocab().most_common(30)]
freqs = [words.vocab()[sample] for sample in samples]
plt.bar(samples, freqs)
plt.xticks(rotation=90)
plt.show()

# 학습 데이터의 단어들로 word_cloud 그리기
wc = WordCloud(width=1000, height=600, background_color="white", font_path=font_path)
plt.imshow(wc.generate_from_frequencies(words.vocab()))
plt.axis("off")
plt.show()