from common.util import cos_similarity, most_similar, analogy
import pickle

pkl_file = 'cbow_params.pkl'
# pkl_file = 'skipgram_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

# print(cos_similarity(word_to_id['boy'], word_to_id['girl']))
# print(cos_similarity(word_to_id['mom'], word_to_id['parent']))

# most similar task
# querys = ['you', 'year', 'car', 'toyota']
querys = ['adult', 'boy', 'child']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# analogy task
print('-'*50)
analogy('king', 'man', 'queen',  word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs, answer='worse')

analogy('boy', 'male', 'girl',  word_to_id, id_to_word, word_vecs, answer='female')