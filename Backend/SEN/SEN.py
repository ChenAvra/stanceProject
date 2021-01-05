#!/usr/bin/env python
# coding: utf-8

# In[9]:


import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import wordninja
import re
import json
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.metrics import accuracy_score
import os


# In[2]:

def get_legal_directort_name(old_name):
    new_name = old_name.replace("/","")
    new_name = new_name.replace("\\", "")
    new_name = new_name.replace(":", "")
    new_name = new_name.replace("*", "")
    new_name = new_name.replace("?", "")
    new_name = new_name.replace('"', "")
    new_name = new_name.replace("<", "")
    new_name = new_name.replace(">", "")
    new_name = new_name.replace("|", "")
    return new_name

def split_to_folders_headline_based(train, test):
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    newpath = BASE_DIR+".\\SEN\\topics"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    topicPath = ".\\SEN\\topics\\headline_based"
    if not os.path.exists(topicPath):
        os.makedirs(topicPath)

    txtPath_train = ".\\SEN\\topics\\headline_based\\train.txt"
    new_df_train = train
    new_df_train.to_csv(txtPath_train, index=None, sep='\t')

    txtPath_test = ".\\SEN\\topics\\headline_based\\test.txt"
    new_df_test = test
    new_df_test.to_csv(txtPath_test, index=None, sep='\t')

def split_to_topic_folders(train, test):
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    newpath = BASE_DIR+"\\topics"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    topic_list=[]
    for topic in train.Claim.unique():
        new_topic = get_legal_directort_name(topic)
        topic_list.append(new_topic)
        #open folder for each topic
        topicPath = BASE_DIR+"\\topics\\"+new_topic
        if not os.path.exists(topicPath):
            os.makedirs(topicPath)

        txtPath_train =BASE_DIR+"\\topics\\"+new_topic+"\\train.txt"
        new_df_train=train.loc[train['Claim'] == topic]
        new_df_train.to_csv(txtPath_train, index=None, sep='\t')

        txtPath_test =BASE_DIR+"\\topics\\"+new_topic+"\\test.txt"
        new_df_test=test.loc[test['Claim'] == topic]
        new_df_test.to_csv(txtPath_test, index=None, sep='\t')

    return topic_list

stemmer = PorterStemmer()
def union_feature_extraction():
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    arr = os.listdir(BASE_DIR+"\\topics")
    for topic in arr:
        STA_path_train = BASE_DIR+"\\topics\\"+topic+"\\STA_feature_extraction_train.csv"
        ent_path_train = BASE_DIR+"\\topics\\"+topic+"\\ent_feature_extraction_train.csv"
        senti_path_train = BASE_DIR+"\\topics\\"+topic+"\\senti_feature_extraction_train.csv"
        STA_path_test = BASE_DIR+"\\topics\\" + topic + "\\STA_feature_extraction_test.csv"
        ent_path_test = BASE_DIR+"\\topics\\" + topic + "\\ent_feature_extraction_test.csv"
        senti_path_test = BASE_DIR+"\\topics\\" + topic + "\\senti_feature_extraction_test.csv"
        train_path = BASE_DIR+"\\topics\\" + topic + "\\train.txt"
        test_path = BASE_DIR+"\\topics\\" + topic + "\\test.txt"

        df_STA_train = pd.read_csv(STA_path_train, header=0)
        del df_STA_train['sentence']
        df_ent_train = pd.read_csv(ent_path_train, header=0)
        del df_ent_train['Text']
        del df_ent_train['hypotheses']
        del df_ent_train['result']
        df_senti_train = pd.read_csv(senti_path_train, header=0)
        del df_senti_train['sentence']
        del df_senti_train['positive']
        del df_senti_train['negative']
        del df_senti_train['neutral']

        df_STA_test = pd.read_csv(STA_path_test, header=0)
        del df_STA_test['sentence']
        df_ent_test = pd.read_csv(ent_path_test, header=0)
        del df_ent_test['Text']
        del df_ent_test['hypotheses']
        del df_ent_test['result']
        df_senti_test = pd.read_csv(senti_path_test, header=0)
        del df_senti_test['sentence']
        del df_senti_test['positive']
        del df_senti_test['negative']
        del df_senti_test['neutral']

        df_train = pd.read_csv(train_path, sep="\t", header=0)
        df_train = df_train[['Stance','Sentence']].copy()


        df_test = pd.read_csv(test_path, sep="\t", header=0)
        df_test = df_test [['Stance','Sentence']].copy()

        final_train_df = [df_STA_train,df_ent_train,df_senti_train,df_train]
        final_train_df=pd.concat(final_train_df, axis=1)

        final_test_df = [df_STA_test,df_ent_test,df_senti_test,df_test]
        final_test_df=pd.concat(final_test_df,axis=1)

        featuresPath = BASE_DIR+"\\final_feature_set"
        if not os.path.exists(featuresPath):
            os.makedirs(featuresPath)
        csvPath_train = BASE_DIR+"\\final_feature_set\\{}_train.csv".format(topic)
        final_train_df.to_csv(csvPath_train, index=None)
        csvPath_test = BASE_DIR+"\\final_feature_set\\{}_test.csv".format(topic)
        final_test_df.to_csv(csvPath_test, index=None)





def load_glove_embeddings_set():
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    word2emb = []
    print("start to load data")
    WORD2VEC_MODEL = BASE_DIR+"/glove.6B.300d.txt"
    fglove = open(WORD2VEC_MODEL,encoding="utf8")
    print("opened data file")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        word2emb.append(word)
    fglove.close()
    print("finish to load data")
    return set(word2emb)

def create_normalise_dict(no_slang_data = "\\noslang_data.json", emnlp_dict = "\\emnlp_dict.txt"):
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    print("Creating Normalization Dictionary")
    with open(BASE_DIR+no_slang_data, "r") as f:
        data1 = json.load(f)

    data2 = {}

    with open(BASE_DIR+emnlp_dict,"r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()

    normalization_dict = {**data1,**data2}
    #print(normalization_dict)
    return normalization_dict


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10,100 ]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = [{'C': Cs, 'gamma' : gammas , 'kernel' : ['rbf']},{'C': Cs , 'gamma' : gammas , 'kernel' : ['linear']}]
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[21]:


def train(topic):
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    train_dataset=pd.read_csv(BASE_DIR+"\\final_feature_set\\{}_train.csv".format(topic))
    train_dataset.sentiment[train_dataset.sentiment=='Neutral']=0
    train_dataset.sentiment[train_dataset.sentiment == 'Positive'] = 1
    train_dataset.sentiment[train_dataset.sentiment == 'Negative'] = 2
    y_train=train_dataset['Stance'].copy()
    del train_dataset['Stance']
    del train_dataset['Sentence']
    del train_dataset['pmi_FAVOR']
    del train_dataset['pmi_AGAINST']
    del train_dataset['pmi_NONE']


    test_dataset=pd.read_csv(BASE_DIR+"\\final_feature_set\\{}_test.csv".format(topic))
    test_dataset.sentiment[test_dataset.sentiment=='Neutral']=0
    test_dataset.sentiment[test_dataset.sentiment == 'Positive'] = 1
    test_dataset.sentiment[test_dataset.sentiment == 'Negative'] = 2
    y_test=test_dataset['Stance'].copy()
    del test_dataset['Stance']
    del test_dataset['Sentence']
    del test_dataset['pmi_FAVOR']
    del test_dataset['pmi_AGAINST']
    del test_dataset['pmi_NONE']
    
    best_params = svc_param_selection(train_dataset,y_train,nfolds=5)
    print(best_params)
    if best_params['kernel'] == 'rbf':
        model = svm.SVC(kernel='rbf' ,C = best_params['C'], gamma = best_params['gamma'],probability=True)
    else:
        model = svm.SVC(kernel='linear' ,C = best_params['C'],probability=True)
    
    
    model.fit(train_dataset,y_train)
    
    y_pred = model.predict(test_dataset)

    return y_pred,y_test



from .preprocessing import run_preprocessing
from .STA_feature_extraction import run_STA_feature_extraction
from .te_f import run_te_f_feature_extraction
from .sentiment_api_2 import run_sentiment_feature_extraction


def train_model_topic_based(df_train, df_test, labels, num_of_labels):
    topic_list = split_to_topic_folders(df_train,df_test)
    run_preprocessing()
    run_STA_feature_extraction(labels)
    run_te_f_feature_extraction()
    run_sentiment_feature_extraction()
    union_feature_extraction()
    y_pred,y_test = [],[]
    for dataset in topic_list:
        a,b = train(dataset)
        y_pred.extend(a)
        y_test.extend(b)
        # print(len(a),len(b))
        print("accuracy for topic ", dataset)
        print(accuracy_score(a,b))
        print("***********************")
    import shutil
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    shutil.rmtree(BASE_DIR+'\\topics')
    shutil.rmtree(BASE_DIR +'\\final_feature_set')

    return y_test,y_pred

def train_model_headline_based(df_train, df_test, labels, num_of_labels):
    split_to_folders_headline_based(df_train,df_test)
    run_preprocessing()
    run_STA_feature_extraction(labels)
    run_te_f_feature_extraction()
    run_sentiment_feature_extraction()
    union_feature_extraction()
    y_pred, y_test = [], []
    a, b = train("headline_based")
    y_pred.extend(a)
    y_test.extend(b)
    # print(len(a),len(b))
    print("accuracy for topic ", "headline_based")
    print(accuracy_score(a, b))
    print("***********************")
    import shutil
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    shutil.rmtree(BASE_DIR+'\\topics')
    shutil.rmtree(BASE_DIR +'\\final_feature_set')
    return y_test, y_pred

