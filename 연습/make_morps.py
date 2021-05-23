from common.util import make_morps_sentences

if __name__ == '__main__':
    text = [
        'ㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴ',
        'ㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇㄹㅁㄴㅇ',
    ]
    morps_sentences = make_morps_sentences(text)
    print(morps_sentences)
exit()


# def morps_sentences_append_Kkma_pos(sentence, morps_sentences):
    # morps_sentences.append(Kkma().pos(Okt().normalize(sentence)))

# class KkmaDelayError(Exception):
    # def __init__(self, sentence):
        # super().__init__(f'Error: Kkma가 형태소 분해하는데 너무 오래걸림\n분해하려는 문장->{sentence}')


# proc = mult.Process(target=morps_sentences_append_Kkma_pos, args=(sentence, morps_sentences))
# proc.start()
# print('\n', morps_sentences)
# start_decomposition_time = t.time()
# while True:
#     t.sleep(0.5)
#     print(proc)
#     if not proc.is_alive():
#         print('break ================================')
#         break
#     if t.time()-start_decomposition_time > 10:
#         print('time')
#         proc.terminate()
#         proc.join()
#         # raise KkmaDelayError
#         break

# except KkmaDelayError:
#     morps_sentences.append(Komoran().pos(Okt().normalize(sentence)))
#     print(morps_sentences)
# except:
#     print(f"Error: 분해 오류\n오류 문장->{sentence}")

# def delete_repeat_latter(sentence):
    # target = sentence[0]
    # index = -1
    # target_idxs = []
    # for _ in range(5):
    #     index = sentence.find(target, index + 1)
    #     if index == -1:
    #         break
    #     target_idxs.append(index)

    # if len(target_idxs) > 3:
    #     idx_diff = target_idxs[1] - target_idxs[0]
    #     pre_idx = target_idxs[1]

    #     iter = 1
    #     for idx in enumerate(target_idxs[2:]):
    #         if idx_diff == idx - pre_idx:
    #             iter += 1
    #         else:
    #             iter = 1
    #             idx_diff = idx - pre_idx
    #         pre_idx = idx

    #     if iter > 2:
    #         #+ 최대로 같은 idx차이가 반복된 횟수 저장하기
    # else:
    #     return sentence