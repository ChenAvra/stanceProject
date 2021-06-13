#!/usr/bin/env python
# coding: utf-8
import json
import os
import glob
import sys
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer,WordNetLemmatizer
import re
import wordninja
import numpy as np
import csv
import argparse
import os
import shutil
import copy
#

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


def split(word, word2emb):
    if word in word2emb:
        return [word]
    return wordninja.split(word)

def load_glove_embeddings():
    word2emb = {}
    PROJECT_ROOT = os.path.abspath('__file__')
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    WORD2VEC_MODEL = BASE_DIR + '\\TAN\\glove.6B.300d.txt'
    fglove = open(WORD2VEC_MODEL,encoding="utf8")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        embedding = np.array(cols[1:],dtype="float32")
        word2emb[word]=embedding
    fglove.close()
    return word2emb



sentences_new = []


def preProcessing(train_x,cm_path1,cm_path2,word2emb):
    wnl = WordNetLemmatizer()
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Creating Normalization Dictionary
    with open(cm_path1, "r") as f:
        data1 = json.load(f)

    data2 = {}
    with open(cm_path2, "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()

    normalization_dict = {**data1, **data2}
    word2emb = word2emb
    n_count = 0
    s_words = 0
    new_lines = []
    old_lines = []
    sentences_new=[]
    for i in range(len(train_x)):
        train_before_Pro = train_x[i]
        sent = clean_str(train_before_Pro)
        word_tokens = sent.split(' ')
        normalized_tokens = []
        # normalization
        for word in word_tokens:
            if word in normalization_dict.keys():
                normalized_tokens.append(normalization_dict[word])
                n_count += 1
            else:
                normalized_tokens.append(word)

        # Word Ninja Splitting
        normalized_tokens_s = []
        for word in normalized_tokens:
            normalized_tokens_s.extend(split(word, word2emb))

        final_tokens = normalized_tokens_s

        # Stop Word Removal
        filtered_tokens = []
        for w in normalized_tokens_s:
            if w not in stop_words:
                filtered_tokens.append(w)
            else:
                s_words += 1

        # Stemming using Porter Stemmer
        stemmed_tokens = []
        for w in filtered_tokens:
            stemmed_tokens.append(ps.stem(w))
        final_tokens = stemmed_tokens

        new_sent = ' '.join(final_tokens)
        new_sent=new_sent.split(' ')
        sentences_new.append(new_sent)

    return sentences_new
