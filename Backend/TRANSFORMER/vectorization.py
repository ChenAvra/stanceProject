import os
import numpy as np


## Change the below for usage of higher dimensional embeddings
PROJECT_ROOT = os.path.abspath('__file__')
BASE_DIR = os.path.dirname(PROJECT_ROOT)
GLOVE_DIR = BASE_DIR+"/TRANSFORMER/embeddings/"
GLOVE_FILE = 'glove.6B.50d.txt'
EMBEDDING_DIM = 50
from gensim.models import KeyedVectors
def getEmbeddingWeightsGlove(word_index):
    '''loads embedding and returns numpy weight matrix of shape totalwords+1, embedding dimensions'''

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR,GLOVE_FILE),encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]

        coefs = np.array([float(value) for value in values[1:]])
        embeddings_index[word] = coefs

        # coefs = np.asarray(values[1:], dtype='float32')
        # embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
