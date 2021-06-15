import csv
import copy
import os

import numpy as np
import re
import itertools
from collections import Counter,defaultdict
import torch
import json
from collections import Counter
import wordninja
from sklearn.model_selection import train_test_split

from .preprocessing import preProcessing



"""

Tokenization/string cleaning for all datasets.
Every dataset is lower cased.
Original taken from https://github.com/dennybritz/cnn-text-classification-tf

string = re.sub(r"[^A-Za-z0-9(),!?\'\`#]", " ", string)
string = re.sub(r"#SemST", "", string)
string = re.sub(r"#([A-Za-z0-9]*)", r"# \1 #", string)
#string = re.sub(r"# ([A-Za-z0-9 ]*)([A-Z])(.*) #", r"# \1 \2\3 #", string)
string =  re.sub(r"([A-Z])", r" \1", string)
string = re.sub(r"\'s", " \'s", string)
string = re.sub(r"\'ve", " \'ve", string)
string = re.sub(r"n\'t", " n\'t", string)
string = re.sub(r"\'re", " \'re", string)
string = re.sub(r"\'d", " \'d", string)
string = re.sub(r"\'ll", " \'ll", string)
string = re.sub(r",", " , ", string)
string = re.sub(r"!", " ! ", string)
string = re.sub(r"\(", " ( ", string)
string = re.sub(r"\)", " ) ", string)
string = re.sub(r"\?", " ? ", string)
string = re.sub(r"\s{2,}", " ", string)
return string.strip() if TREC else string.strip().lower()

"""
def clean_str2(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets.
    Every dataset is lower cased.
    Original taken from https://github.com/dennybritz/cnn-text-classification-tf
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()



def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets.
    Every dataset is lower cased.
    Original taken from https://github.com/dennybritz/cnn-text-classification-tf
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`#]", " ", string)
    string = re.sub(r"#SemST", "", string)
    string = re.sub(r"#([A-Za-z0-9]*)", r"# \1 #", string)
    #string = re.sub(r"# ([A-Za-z0-9 ]*)([A-Z])(.*) #", r"# \1 \2\3 #", string)
    #string =  re.sub(r"([A-Z])", r" \1", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def create_normalise_dict(no_slang_data = "./noslang_data.json", emnlp_dict = "./emnlp_dict.txt"):
    print("Creating Normalization Dictionary")
    with open(no_slang_data, "r") as f:
        data1 = json.load(f)

    data2 = {}

    with open(emnlp_dict,"r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()

    normalization_dict = {**data1,**data2}
    #print(normalization_dict)
    return normalization_dict

def normalise(normalization_dict,sentence):
    normalised_tokens = []
    word_tokens = sentence.split()
    for word in word_tokens:
        if word in normalization_dict:
        #if False:
            normalised_tokens.extend(normalization_dict[word].lower().split(" "))
            #print(word," normalised to ",normalization_dict[word])
        else:
            normalised_tokens.append(word.lower())
    #print(normalised_tokens)
    return normalised_tokens


def load_dataset(topic_string,df_train,df_test,labels,dataset,dev = "cuda"):
    def split(word):
        if word in word2emb:
        #if True:
            return [word]
        return wordninja.split(word)

    assert dataset in ['Are E-Cigarettes safe?', 'Does Sunlight exposure lead to skin cancer?',
                       'Does Vitamin C prevent common cold?', 'Should women take HRT post-menopause?',
                       'Does MMR Vaccine lead to autism in children?', 'Atheism', 'Hillary Clinton',
                       'Legalization of Abortion', 'Climate Change is a Real Concern',
                       'Feminist Movement'], "unknown dataset"

    folder = "Data_SemE_P"

    if dataset == 'Are E-Cigarettes safe?':
        topic = 'E-ciggarettes are safer than normal ciggarettes'
        folder = "Data_MPCHI_P"
    elif dataset == 'Does Sunlight exposure lead to skin cancer?':
        topic = 'Sun exposure can lead to skin cancer'
        folder = "Data_MPCHI_P"
    elif dataset == 'Does Vitamin C prevent common cold?':
        topic = 'Vitamin C prevents common cold'
        folder = "Data_MPCHI_P"
    elif dataset == 'Should women take HRT post-menopause?':
        topic = 'Women should take HRT post menopause'
        folder = "Data_MPCHI_P"
    elif dataset == 'Does MMR Vaccine lead to autism in children?':
        topic = 'MMR vaccine can cause autism'
        folder = "Data_MPCHI_P"
    elif dataset == 'Atheism':
        topic = "Atheism"
    elif dataset == 'Hillary Clinton':
        topic = "Hillary Clinton"
    elif dataset == 'Legalization of Abortion':
        topic = "Legalization of Abortion"
    elif dataset == 'Climate Change is a Real Concern':
        topic = "Climate Change is a Real Concern"
    elif dataset == 'Feminist Movement':
        topic = "Feminist Movement"
    elif dataset == 'vaccines cause autism':
        topic = "vaccines cause autism"
    elif dataset == 'vaccines treat influenza':
        topic = "vaccines treat influenza"
    else:
        topic = topic_string
    print(topic)
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    cm_path1 = BASE_DIR + '\\noslang_data.json'
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    cm_path2 = BASE_DIR + '\\emnlp_dict.txt'
    normalization_dict = create_normalise_dict(no_slang_data = cm_path1, emnlp_dict = cm_path2)

    target = normalise(normalization_dict,clean_str(topic))

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

    # stances = {'FAVOR' : 0, 'AGAINST' : 1, 'NONE' : 2}
    stances={}

    for i in range(len(labels)):
        stances.update({labels[i] : i})


    train_x_before_split = []
    train_y_before_split = []
    #iter train
    for index, row in df_train.iterrows():
        train_before_Pro = row['Sentence']
        stances_before_Pro = row['Stance']
        # data = normalise(normalization_dict, clean_str(train_before_Pro))
        train_x_before_split.append(train_before_Pro)
        train_y_before_split.append(stances[row['Stance']])

    # sentences_new_train = preProcessing(train_x_before_split,cm_path1,cm_path2)

    test_x_before_proc=[]
    test_y_before_proc=[]
    #iter test
    for index, row in df_test.iterrows():
        train_before_Pro = row['Sentence']
        stances_before_Pro = row['Stance']
        # data = normalise(normalization_dict, clean_str(train_before_Pro))
        test_x_before_proc.append(train_before_Pro)
        test_y_before_proc.append(stances[row['Stance']])

    train_x=preProcessing(train_x_before_split,cm_path1,cm_path2,word2emb)
    train_y=train_y_before_split
    test_x=preProcessing(test_x_before_proc,cm_path1,cm_path2,word2emb)
    test_y=test_y_before_proc


    word_ind = {}



    for sent in train_x:
        for word in sent:
            if word not in word_ind and word in word2emb:
                word_ind[word] = len(word_ind)

    for sent in test_x:
        for word in sent:
            if word not in word_ind and word in word2emb:
                word_ind[word] = len(word_ind)

    for word in target:
        if word not in word_ind and word in word2emb:
            word_ind[word] = len(word_ind)



    UNK = len(word_ind)
    PAD = len(word_ind)+1


    ind_word = {v:k for k,v in word_ind.items()}


    print("Number of words - {}".format(len(ind_word)))


    # In[12]:





    x_train = []
    OOV = 0
    oovs = []

    for i,sent in enumerate(train_x):
        temp = []
        for j,word in enumerate(sent):
            if word in word_ind:
                temp.append(word_ind[word])
            else:
                #print(word)
                temp.append(UNK)
                OOV+=1
                oovs.append(word)
        x_train.append(temp)

    print("OOV words :- ",OOV)
    a = Counter(oovs)
    print(a)



    y_train = np.array(train_y)
    y_test = np.array(test_y)




    x_test = []

    for i,sent in enumerate(test_x):
        temp = []
        for j,word in enumerate(sent):
            if word in word_ind:
                temp.append(word_ind[word])
            else:
                temp.append(UNK)

        x_test.append(temp)




    embedding_matrix = np.zeros((len(word_ind) + 2, 300))
    embedding_matrix[len(word_ind)] = np.random.randn((300))
    for word in word_ind:
        embedding_matrix[word_ind[word]] = word2emb[word]




    print("Number of training examples :- ",len(x_train))
    print("Sample vectorised sentence :- ",x_train[0])

    device = torch.device(dev)
    print("Using this device :- ", device)





    vector_target = []
    for w in target:
        if w in word_ind:
            vector_target.append(word_ind[w])
        else:
            vector_target.append(UNK)


    print("vectorised target:-")
    print(vector_target)

    return stances, word2emb, word_ind, ind_word, embedding_matrix, device,\
     x_train, y_train, x_test, y_test, vector_target, train_x, test_x



def pre_proce_one_stance(word_ind_load,topic_string,df,labels,dataset,dev = "cuda"):
    def split(word):
        if word in word2emb:
            # if True:
            return [word]
        return wordninja.split(word)



    folder = "Data_SemE_P"
    topic_string=''
    if dataset == 'Are E-Cigarettes safe?':
        topic = 'E-ciggarettes are safer than normal ciggarettes'
        folder = "Data_MPCHI_P"
    elif dataset == 'Does Sunlight exposure lead to skin cancer?':
        topic = 'Sun exposure can lead to skin cancer'
        folder = "Data_MPCHI_P"
    elif dataset == 'Does Vitamin C prevent common cold?':
        topic = 'Vitamin C prevents common cold'
        folder = "Data_MPCHI_P"
    elif dataset == 'Should women take HRT post-menopause?':
        topic = 'Women should take HRT post menopause'
        folder = "Data_MPCHI_P"
    elif dataset == 'Does MMR Vaccine lead to autism in children?':
        topic = 'MMR vaccine can cause autism'
        folder = "Data_MPCHI_P"
    elif dataset == 'Atheism':
        topic = "Atheism"
    elif dataset == 'Hillary Clinton':
        topic = "Hillary Clinton"
    elif dataset == 'Legalization of Abortion':
        topic = "Legalization of Abortion"
    elif dataset == 'Climate Change is a Real Concern':
        topic = "Climate Change is a Real Concern"
    elif dataset == 'Feminist Movement':
        topic = "Feminist Movement"
    elif dataset == 'vaccines cause autism':
        topic = "vaccines cause autism"
    elif dataset == 'vaccines treat influenza':
        topic = "vaccines treat influenza"
    else:
        topic = topic_string
    print(topic)
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    cm_path1 = BASE_DIR + '\\noslang_data.json'
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    cm_path2 = BASE_DIR + '\\emnlp_dict.txt'
    normalization_dict = create_normalise_dict(no_slang_data=cm_path1, emnlp_dict=cm_path2)

    target = normalise(normalization_dict, clean_str(topic))

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

    stances = {}

    for i in range(len(labels)):
        stances.update({labels[i]: i})

    test_x_before_split = []
    test_y_before_split = []
    # iter train
    for index, row in df.iterrows():
        train_before_Pro = row['Sentence']
        stances_before_Pro = row['Stance']
        # data = normalise(normalization_dict, clean_str(train_before_Pro))
        test_x_before_split.append(train_before_Pro)
        test_y_before_split.append(stances[row['Stance']])




    test_x = preProcessing(test_x_before_split, cm_path1, cm_path2, word2emb)
    train_y = test_y_before_split

    word_ind = word_ind_load





    for sent in test_x:
        for word in sent:
            if word not in word_ind and word in word2emb:
                word_ind[word] = len(word_ind)

    for word in target:
        if word not in word_ind and word in word2emb:
            word_ind[word] = len(word_ind)

    UNK = len(word_ind)
    PAD = len(word_ind) + 1

    ind_word = {v: k for k, v in word_ind.items()}

    print("Number of words - {}".format(len(ind_word)))

    OOV = 0
    oovs = []



    print("OOV words :- ", OOV)
    a = Counter(oovs)
    print(a)



    x_test = []

    for i, sent in enumerate(test_x):
        temp = []
        for j, word in enumerate(sent):
            if word in word_ind:
                temp.append(word_ind[word])
            else:
                temp.append(UNK)

        x_test.append(temp)

    embedding_matrix = np.zeros((len(word_ind) + 2, 300))
    embedding_matrix[len(word_ind)] = np.random.randn((300))
    for word in word_ind:
        embedding_matrix[word_ind[word]] = word2emb[word]



    device = torch.device(dev)
    print("Using this device :- ", device)

    vector_target = []
    for w in target:
        if w in word_ind:
            vector_target.append(word_ind[w])
        else:
            vector_target.append(UNK)

    print("vectorised target:-")
    print(vector_target)

    return  x_test, vector_target


def load_glove_embeddings():
    word2emb = {}
    WORD2VEC_MODEL = "../Preprocessing/glove.6B.300d.txt"
    fglove = open(WORD2VEC_MODEL,encoding="utf8")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        embedding = np.array(cols[1:],dtype="float32")
        word2emb[word]=embedding
    fglove.close()
    return word2emb
