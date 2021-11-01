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
data_file = r'ratings.txt'

with open('saved_pkls/YT_cmts_morps_to_id_Kkma.pkl', 'rb') as f:
    (corpus, morp_to_id, id_to_morp) = pickle.load(f)
# with open('saved_pkls/YT_cmts2_morps_to_id_Kkma.pkl', 'rb') as f:
#     (corpus2, morp_to_id, id_to_morp) = pickle.load(f)
# np.concatenate((corpus, corpus2))

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

# corpus = [id_to_morp[corp] for corp in corpus]
print('\n명사만 추출 중...')
words = Text(nouns(corpus, morp_to_id, id_to_morp), name="morps")

words.plot(30)
plt.show()

# 학습 데이터의 단어들로 word_cloud 그리기
wc = WordCloud(width=1000, height=600, background_color="white", font_path=font_path)
plt.imshow(wc.generate_from_frequencies(words.vocab()))
plt.axis("off")
plt.show()