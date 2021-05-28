from random import random

from Backend.TAN.early_stopping_training import run_model, pred_one_stance,train_bagging_tan_CV,load_dataset
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
labels_df = get_unique_labels(df)
num_of_labels = len(labels_df)
topics = df.Claim.unique()
num_of_topics = len(topics)



import pandas as pd
def test_run_model():


    labels_pred,y_test,len_ensemble_model,labels,embedding_matrix,word_ind=run_model(df_train,df_test,labels_df,num_of_labels)
    is1=isinstance(labels_pred,list)
    is2=isinstance(y_test,list)


def test_get_predict_per_stance_test():
    claim = "Hilary Clinton"
    sentence = "I love hialry clinton"
    labels_pred,y_test,len_ensemble_model,labels,embedding_matrix,word_ind=run_model(df_train,df_test,labels_df,num_of_labels)
    label=pred_one_stance( labels,embedding_matrix,sentence,claim,None, word_ind)
    isinstance(label,label in labels)

def test_train_bagging_tan_CV():

    topic_string=''
    claim="Atheism"
    stances, word2emb, word_ind, ind_word, embedding_matrix, device, \
    x_train, y_train, x_test, y_test, vector_target, train_tweets, test_tweets = load_dataset(topic_string, df_train,df_test, labels_df, claim,
                                                                                              dev="cpu")

    combined = list(zip(x_train, y_train))
    random.shuffle(combined)
    x_train[:], y_train[:] = zip(*combined)

    labels_pred,y_test,len_ensemble_model,labels,embedding_matrix,word_ind=train_bagging_tan_CV(stances,x_train, y_train, x_test, y_test, vector_target,labels_df,device,embedding_matrix,claim,version="tan-",n_epochs=1,batch_size=50,l2=0,dropout = 0.5,n_folds=2)
    is1=isinstance(labels_pred,list)
    is2=isinstance(y_test,list)

