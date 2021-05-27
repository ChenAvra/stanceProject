import os

from Backend.TAN.preprocessing import clean_str,preProcessing,load_glove_embeddings
import numpy as np
from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels
from Backend.DB.DBManager import DataBase
db = DataBase()
dataset_name = "semEval2016"
dataset_id = dataset_names_dict[dataset_name]
df = db.get_dataset(dataset_id)
df_train, df_test = split_data_topic_based(df, 0.8)
df_train_records = df_train.shape[0]
df_test_records = df_test.shape[0]
type = 'topic'
labels = get_unique_labels(df)
num_of_labels = len(labels)
topics = df.Claim.unique()
num_of_topics = len(topics)
def test_clean_str():
    sentence="Hilary Clinton"

    string=clean_str(sentence)
    s1=isinstance(string,str)

    assert (s1,True)

def test_preprocessing():
    word2emb = {}
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    WORD2VEC_MODEL = BASE_DIR + '\\glove.6B.300d.txt'
    fglove = open(WORD2VEC_MODEL, encoding="utf8")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        embedding = np.array(cols[1:], dtype="float32")
        word2emb[word] = embedding
    fglove.close()

    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    cm_path1 = BASE_DIR + '\\noslang_data.json'
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    cm_path2 = BASE_DIR + '\\emnlp_dict.txt'

    stances = {}

    for i in range(len(labels)):
        stances.update({labels[i]: i})

    train_x_before_split = []
    train_y_before_split = []
    # iter train
    for index, row in df_train.iterrows():
        train_before_Pro = row['Sentence']
        stances_before_Pro = row['Stance']
        # data = normalise(normalization_dict, clean_str(train_before_Pro))
        train_x_before_split.append(train_before_Pro)
        train_y_before_split.append(stances[row['Stance']])

    train=preProcessing(train_x_before_split,cm_path1,cm_path2,word2emb)
    is1=isinstance(train,list)
    assert (is1,True)


def test_load_glove_embeddings():

    matrix=load_glove_embeddings()
    s1=isinstance(matrix,dict)

    assert (s1,True)


