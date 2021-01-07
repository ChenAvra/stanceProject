#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 03:05:06 2018

@author: shalmoli
"""
import pandas as pd
import csv
from pycorenlp import StanfordCoreNLP
import json
import os




#storing Result
# before Executing this,we have to start the Stanford Server through Terminal
def sentiment_feature_extraction(pathToRead, pathToWrite):
    # reading Input file
    data = pd.read_csv(pathToRead, header=0)
    error_count=0
    # data = pd.read_csv(
    #     'C:\\Users\\iris dreizenshtok\\Desktop\\programming\\Stance-Detection-in-Web-and-Social-Media-master\\SEN-SEN\\Data_MPCHI\\HRT\\test.txt',
    #     sep="\t", header=0)
    # Extracting Sentences
    sentence = data['Sentence']
    length = len(sentence)
    count_=0
    '''It's returns sentiment sentences wise.
    If we have have abstract more than one sentence ,we conclude the sentiment for that sentence by considering the majority and breaking tie randomly.'''
    result = []
    print("working now on:", pathToRead)
    for i in range(length):
        nlp = StanfordCoreNLP('http://localhost:9000')
        res = nlp.annotate(sentence[i],
                           properties={
                               'annotators': 'sentiment',
                               'outputFormat': 'json',
                               'timeout': 100000,
                           })
        count_ = count_ + 1
        # print(count_)
        # print("sentence is:"+sentence[i])
        count = 0  # counting number of sentences in input
        count_1 = 0  # counting Negative Sentiment
        count_2 = 0  # counting Neutral Sentiment
        count_3 = 0  # counting Positive Sentiment
        if(res != 'Could not handle incoming annotation' and res != 'CoreNLP request timed out. Your document may be too long.'):
            for s in (res['sentences']):
                if (s["sentimentValue"]):
                    count = count + 1
                    if (s["sentiment"] == 'Negative' or s["sentiment"] == 'Verynegative'):
                        count_1 = count_1 + 1
                    else:
                        if (s["sentiment"] == 'Positive' or s["sentiment"] == 'Verypositive'):
                            count_3 = count_3 + 1
                        else:
                            count_2 = count_2 + 1
            if (count > 1):
                if (count_1 > count_2 and count_1 > count_3):
                    result.append("Negative")
                else:
                    if (count_2 > count_1 and count_2 > count_3):
                        result.append('Neutral')
                    else:
                        result.append('Positive')
            else:
                if (s["sentiment"] == 'Negative' or s["sentiment"] == 'Verynegative'):
                    result.append("Negative")
                else:
                    if (s["sentiment"] == 'Positive' or s["sentiment"] == 'Verypositive'):
                        result.append("Positive")
                    else:
                        result.append('Neutral')
        else:
            result.append('Neutral')
            error_count=error_count+1
    # print("error count is:", error_count)
    # Storing Output to file    Oytput is binary for each class individually
    file = open(pathToWrite, 'w', encoding='utf-8')
    fields = ('sentence', 'positive', 'negative', 'neutral', 'sentiment')
    wr = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
    wr.writeheader()
    for i in range(length):
        if (result[i] == 'Negative'):
            wr.writerow({'sentence': sentence[i], 'positive': 0, 'negative': 1, 'neutral': 0, 'sentiment': result[i]})
        else:
            if (result[i] == 'Positive'):
                wr.writerow(
                    {'sentence': sentence[i], 'positive': 1, 'negative': 0, 'neutral': 0, 'sentiment': result[i]})
            else:
                wr.writerow(
                    {'sentence': sentence[i], 'positive': 0, 'negative': 0, 'neutral': 1, 'sentiment': result[i]})

    file.close()

def run_sentiment_feature_extraction():
    import os
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    os.popen('java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000')
    arr = os.listdir(BASE_DIR+"\\topics")
    for topic in arr:
        for k in ["train","test"]:
            csvFileToRead=BASE_DIR+"\\topics\\"+topic+"\\"+k+".csv"
            outputFileToWrite = BASE_DIR+"\\topics\\"+topic+"\\senti_feature_extraction_"+k+".csv"
            sentiment_feature_extraction(csvFileToRead,outputFileToWrite)

