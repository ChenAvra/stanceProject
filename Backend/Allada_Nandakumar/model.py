import gensim
import numpy as np
import os
import re
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from gensim.models import KeyedVectors
from keras import callbacks
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Bidirectional
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate,dot
from keras.optimizers import Adam
from keras.utils import np_utils
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith

def text_cleaner(text):
    stop_words = set(stopwords.words('english'))
    newString = text.lower()
    newString = re.sub('<br\s?\/>|<br>', "", newString)
    newString = newString.replace('\n', '')
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)

    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)

    tokens = [w for w in newString.split() if not w in stop_words]
    return (" ".join(tokens)).strip()

def Pred(df_train, df_test, l):
    LABELS = l.tolist()
    combine_df_train = df_train.copy()
    combine_df_test = df_test.copy()
    combine_df_train['Stance'] = combine_df_train['Stance'].apply(lambda x: LABELS.index(x))
    combine_df_test['Stance'] = combine_df_test['Stance'].apply(lambda x: LABELS.index(x))


    # Specify the folder locations
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    W2V_DIR = BASE_DIR + '\\GoogleNews-vectors-negative300.bin'  # W2v
    #GloVe_DIR = path + '\\glove.6B.300d.txt'  # Glove
    #FastTxt_DIR = path + '\\wiki-news-300d-1M.vec'  # fasttext

    # CONFIG

    # the data directory
    MAX_SENT_LEN = 150  # 75(0.68), 150, 300 700(90% but too time comsuming)
    MAX_VOCAB_SIZE = 40000  # vocabulary
    EMBEDDING_DIM = 300  # 50 for GloVe 300 for w2v


    # PREPROCESS

    train_head = [text_cleaner(head) for head in combine_df_train['Claim']]
    train_body = [text_cleaner(body) for body in combine_df_train['Sentence']]
    test_head = [text_cleaner(head) for head in combine_df_test['Claim']]
    test_body = [text_cleaner(body) for body in combine_df_test['Sentence']]

    # TF-IDF

    vectorizer = TfidfVectorizer(max_features = 150)
    X_train_head_tfidf = vectorizer.fit_transform(train_head).toarray()
    X_train_body_tfidf = vectorizer.fit_transform(train_body).toarray()
    X_test_head_tfidf= vectorizer.transform(test_head).toarray()
    X_test_body_tfidf = vectorizer.transform(test_body).toarray()


    result = np.zeros(X_train_body_tfidf.shape)
    result[:X_train_head_tfidf.shape[0], :X_train_head_tfidf.shape[1]] = X_train_head_tfidf
    X_train_head_tfidf = result


    # Pre-processing involves removal of puctuations and converting text to lower case
    word_seq_head_train = [text_to_word_sequence(text_cleaner(head)) for head in combine_df_train['Claim']]
    word_seq_bodies_train = [text_to_word_sequence(text_cleaner(body)) for body in combine_df_train['Sentence']]
    word_seq_head_test = [text_to_word_sequence(text_cleaner(head)) for head in combine_df_test['Claim']]
    word_seq_bodies_test = [text_to_word_sequence(text_cleaner(body)) for body in combine_df_test['Sentence']]


    word_seq = []
    for i in range(len(word_seq_head_train)):
        word_seq.append(word_seq_head_train[i])
    for i in range(len(word_seq_bodies_train)):
        word_seq.append(word_seq_bodies_train[i])

    for i in range(len(word_seq_head_test)):
        word_seq.append(word_seq_head_test[i])
    for i in range(len(word_seq_bodies_test)):
        word_seq.append(word_seq_bodies_test[i])


    filter_list = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters=filter_list)
    tokenizer.fit_on_texts([seq for seq in word_seq])
    #because it only includes unique words(tokens)

    print("Number of words in vocabulary:", len(tokenizer.word_index))


    # TEXT with two LSTM FOR head and body

    # Shorten the sentence to a fixed length
    # Convert the sequence of words to sequnce of indices
    X_train_head = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq_head_train])
    X_train_head = pad_sequences(X_train_head, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

    X_train_body = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq_bodies_train])
    X_train_body = pad_sequences(X_train_body, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

    y_train_1 = combine_df_train['Stance']

    X_test_head = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq_head_test])
    X_test_head = pad_sequences(X_test_head, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

    X_test_body = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq_bodies_test])
    X_test_body = pad_sequences(X_test_body, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

    y_test_1 = combine_df_test['Stance']

    y_train = np_utils.to_categorical(y_train_1)
    y_test = np_utils.to_categorical(y_test_1)


    # W2v

    def embed_matrix(model,word_index,EMBEDDING_DIM):
      embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_index)+1, EMBEDDING_DIM)) # +1 is because the matrix indices start with 0
      for word, i in word_index.items(): # i=0 is the embedding for the zero padding
          try:
              embeddings_vector = model.wv.get_vector(word)
          except KeyError:
              embeddings_vector = None
              #none: if words in sentence don't have pre-trained corresponding embedding, then error occurs

          if embeddings_vector is not None:
              embeddings_matrix[i] = embeddings_vector
              #if pre-trained word embedding exists，then let embeddings_matrix[i] is this embedding
              #Wi:the ith row of embeddings_matrix

      return embeddings_matrix


    # Load the word2vec embeddings

    embeddings_w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_DIR, binary=True)
    embeddings_matrix_w2v = embed_matrix(embeddings_w2v,tokenizer.word_index,EMBEDDING_DIM)

    head_input = Input(shape=(MAX_SENT_LEN,), dtype='int32', name='head_input')
    body_input = Input(shape=(MAX_SENT_LEN,), dtype='int32', name='body_input')

    shared_embed = Embedding(len(tokenizer.word_index) + 1,EMBEDDING_DIM,weights=[embeddings_matrix_w2v],trainable=False)
    head_embed = shared_embed(head_input)
    body_embed = shared_embed(body_input)

    shared_lstm = Bidirectional(LSTM(100,dropout=0.2, recurrent_dropout=0.2, name='head_lstm'))
    head_lstm = shared_lstm(head_embed)
    body_lstm = shared_lstm(body_embed)

    dot_layer = dot([head_lstm, body_lstm], axes=1, normalize=True)

    head_input_tfidf = Input(shape=(MAX_SENT_LEN,))
    body_input_tfidf = Input(shape=(MAX_SENT_LEN,))

    tf_dense = Dense(100, activation='relu')
    # tf_dense = Dropout(0.4)(tf_dense)
    tf_dense_head = tf_dense(head_input_tfidf)
    tf_dense_body = tf_dense(body_input_tfidf)

    dot_layer_tfidf = dot([tf_dense_head, tf_dense_body], axes=1, normalize=True)

    conc = concatenate([head_lstm, body_lstm, dot_layer, tf_dense_head, tf_dense_body, dot_layer_tfidf])

    dense = Dense(100, activation='relu')(conc)
    dense = Dropout(0.3)(dense)
    dense = Dense(len(LABELS), activation='softmax')(dense)
    model = Model(inputs=[head_input, body_input, head_input_tfidf, body_input_tfidf], outputs=[dense])
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


    # MAIN MODEL

    filepath = BASE_DIR + "\\fraud_1750.hdf5"
    checkpoint = callbacks.ModelCheckpoint(filepath,
                                           monitor='val_accuracy',
                                           verbose=1,mode='max',
                                           save_best_only=True)
    callbacks_list1 = [checkpoint]
    his = model.fit([X_train_head,X_train_body,X_train_head_tfidf,X_train_body_tfidf],[y_train],epochs=1, validation_data=([X_test_head,X_test_body,X_test_head_tfidf,X_test_body_tfidf],[y_test]), batch_size=64,verbose = True,callbacks = callbacks_list1)

    pred_1 = model.predict([X_test_head, X_test_body, X_test_head_tfidf, X_test_body_tfidf])

    # y_prob = model.predict_proba(df_test['Stance'])
    pred_old = np.argmax(pred_1, axis=1)

    # Load/Precompute all features now
    # Run on competition dataset

    actual = df_test['Stance']
    #actual = [LABELS[int(a)] for a in y_competition]
    predict_11 = [LABELS[int(a)] for a in pred_old]
    os.remove(BASE_DIR + "\\fraud_1750.hdf5")

    actual = actual.values.tolist()
    return actual, predict_11, pred_1
