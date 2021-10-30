from __future__ import unicode_literals, print_function, division
from io import open
import random

import torch
from model import EncoderRNN, AttnDecoderRNN
from train_module import trainIters, evaluate, evaluateRandomly 
from train_module import device

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

import pickle

################################################################################################################################

# 학습할 데이터 읽어 학습/검증/테스트 데이터로 나누기
lang_file = 'saved_pkls/test_lang_corpus.pkl'
with open(lang_file, 'rb') as file:
    lang, corpus, sentences = pickle.load(file).values()

input_lang, output_lang = lang, lang
pairs = [[sentence[:-1], sentence[1:]] for sentence in sentences]

max_length_ = 0
for sentence in sentences:
    max_length_ = max(max_length_, len(sentence)-1+1) # -1: [:-1] / +1: EOS_TOKEN
print("MAX_LENGTH =", max_length_)

################################################################################################################################

font_path = r'common/font/주아체.ttf' # 한글 폰트 경로
font_name = fm.FontProperties(fname=font_path, size=50).get_name()
plt.rc('font', family=font_name)
# print(font_name)
# import matplotlib as mpl
# mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 15

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # 주기적인 간격에 이 locator가 tick을 설정
    loc = ticker.MultipleLocator(base=0.5)
    ax.yaxis.set_major_locator(loc)
    plt.xlabel('학습한 문장 수 (100문장)')
    plt.ylabel('Loss (손실)')
    plt.plot(points)
    plt.show()

# 한글 출력 테스트
plt.text(0.2, 0.3, '한글', size=100)
plt.show()

################################################################################################################################

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_morps, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_morps, dropout_p=0.1).to(device)

plot_losses = trainIters(encoder1, attn_decoder1, 10000, pairs, print_every=1000)
torch.save(encoder1, 'torch_save/EncoderRNN.pth')
torch.save(attn_decoder1, 'torch_save/AttnDecoderRNN.pth')
showPlot(plot_losses)

encoder1 = torch.load('torch_save/EncoderRNN.pth')
attn_decoder1 = torch.load('torch_save/AttnDecoderRNN.pth')

evaluateRandomly(encoder1, attn_decoder1, pairs, output_lang, device)

################################################################################################################################

import unicodedata

def showAttention(input_sentence, output_words, attentions):
    input_sentence = unicodedata.normalize('NFC',input_sentence) # 한글 깨짐 방지
    
    # colorbar로 그림 설정
    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    
    # 축 설정
    ax.set_xticklabels([])
    # ax.set_xticklabels([''] + input_sentence.split(' ') +
    #                    ['<EOS>'], rotation=90, fontproperties=font_name)
    ax.set_yticklabels([''] + output_words, fontproperties=font_name)

    for i, word in enumerate(input_sentence.split(' ')+['<EOS>']):
        ax.text(-0.2+i*1, -0.8, word, rotation=90)   
    # for i, word in enumerate(output_words):
    #     ax.text(-0.8, i*1, word, horizontalalignment='right', size=15)

    # 매 틱마다 라벨 보여주기
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # ticks_loc_x = ax.get_xticks().tolist()
    # ticks_loc_y = ax.get_yticks().tolist()
    # ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc_x))
    # ax.yaxis.set_major_locator(ticker.FixedLocator(ticks_loc_y))
    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence, input_lang, output_lang, device)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


import warnings
warnings.filterwarnings("ignore")

evaluateAndShowAttention("그는 내일 오후에 떠날 예정이다 .")

evaluateAndShowAttention("그 사람은 감기로 몸살을 앓고 있어 .")

evaluateAndShowAttention("난 그 사람의 건강이 너무 걱정돼 .")

evaluateAndShowAttention("그는 학급에서 가장 둔한 아이이다 .")


while True:
    a = input()
    if a in ["break", "exit"]: break
    evaluateAndShowAttention(random.choice(pairs)[0])