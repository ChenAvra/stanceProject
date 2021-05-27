import os

from Backend.TAN.utils import create_normalise_dict,normalise,load_dataset,pre_proce_one_stance
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



def test_load_glove_embeddings():

    dictionar=create_normalise_dict(no_slang_data = "./noslang_data.json", emnlp_dict = "./emnlp_dict.txt")
    s1=isinstance(dictionar,dict)

    assert (s1,True)


def test_normalise():
    dict=create_normalise_dict(no_slang_data = "./noslang_data.json", emnlp_dict = "./emnlp_dict.txt")
    sentence="i love holary clinton"
    string=normalise(dict,sentence)
    s1=isinstance(string,list)

    assert (s1,True)


def test_load_dataset():
    stances, word2emb, word_ind, ind_word, embedding_matrix, device, \
    x_train, y_train, x_test, y_test, vector_target, train_x, test_x=load_dataset("Atheism",df_train,df_test,labels,"Atheism")
    s1=isinstance(x_train,list)

    assert (s1,True)
import pandas as pd
def test_pre_proc_one_stance():
    stances, word2emb, word_ind, ind_word, embedding_matrix, device, \
    x_train, y_train, x_test, y_test, vector_target, train_x, test_x = load_dataset("Atheism", df_train, df_test,
                                                                                    labels,"Atheism" )
    Y_test=['AGAINST']
    X_test=["I Love hilary clinton"]
    df2 = pd.DataFrame(list(zip(Y_test, X_test)),
                   columns =['Stance', 'Sentence'])
    x_test, vector_target= pre_proce_one_stance(word_ind,"Atheism",df2,labels,"Atheism")
    s1=isinstance(x_test,list)

    assert (s1,True)



