import numpy as np
from sklearn.utils.extmath import randomized_svd
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

wordvec_size = 100
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)
word_vecs = U[:, :wordvec_size]
np.set_printoptions(precision=3)

# print(C, "\n") # create_co_matrix
# print('id-0 :', C[0])
# print('hello :', C[word_to_id['hello']])

# c0 = C[word_to_id['hello']] # cos_similarity
# c1 = C[word_to_id['goodbye']]
# print(cos_similarity(c0, c1))     

querys = ['you', 'say', 'goodbye', 'and', 'i', 'hello', '.']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# print(C)
# print(W)