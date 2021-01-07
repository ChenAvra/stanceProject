import time
import numpy as np
import pandas as pd
import string
import csv
from scipy import stats
import random
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import wordninja
from collections import defaultdict, Counter
import math
import sys
import nltk
nltk.download('universal_tagset')
import os

def load_glove_embeddings_set():
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    word2emb = []
    WORD2VEC_MODEL = BASE_DIR+"\\glove.6B.300d.txt"
    fglove = open(WORD2VEC_MODEL,encoding="utf8")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        word2emb.append(word)
    fglove.close()
    return set(word2emb)

def create_normalise_dict(no_slang_data = "\\noslang_data.json", emnlp_dict = "\\emnlp_dict.txt"):
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    print("Creating Normalization Dictionary")
    with open(BASE_DIR+no_slang_data, encoding="utf8") as f:
        data1 = json.load(f)

    data2 = {}

    with open(BASE_DIR+emnlp_dict,encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()

    normalization_dict = {**data1,**data2}
    #print(normalization_dict)
    return normalization_dict



def sent_process(sent,word_dict, norm_dict):
    sent=str(sent)
    sent = re.sub(r"[^A-Za-z0-9(),!?\'\`#]", " ", sent)
    sent = re.sub(r"#SemST", "", sent)
    sent = re.sub(r"#([A-Za-z0-9]*)", r"# \1 #", sent)
    #sent = re.sub(r"# ([A-Za-z0-9 ]*)([A-Z])(.*) #", r"# \1 \2\3 #", sent)
    #sent =  re.sub(r"([A-Z])", r" \1", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = re.sub(r",", " , ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent = re.sub(r"\(", " ( ", sent)
    sent = re.sub(r"\)", " ) ", sent)
    sent = re.sub(r"\?", " ? ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    sent = sent.strip()
    word_tokens = sent.split()
    normalised_tokens = []
    for word in word_tokens:
        if word in norm_dict:
        #if False:
            normalised_tokens.extend(norm_dict[word].lower().split(" "))
            #print(word," normalised to ",norm_dict[word])
        else:
            normalised_tokens.append(word.lower())
    wordninja_tokens = []
    for word in normalised_tokens:
        if word in word_dict:
            wordninja_tokens+=[word]
        else:
            wordninja_tokens+=wordninja.split(word)
    return " ".join(wordninja_tokens)


def build_lexicon(labels_array,topic_Path,word_dict, norm_dict):
    def pmi(x, y, z, t):
        res = (x / (y * (z / t) + (math.sqrt(x) * math.sqrt(math.log(0.9) / (-2)))))
        return math.log(res, 2)

    def prob(word1, nava, total):
        count_prob = 0
        if word1 in nava:
            count_prob += nava[word1]
        return ((count_prob + 1))

    def prob_cond(word1, seed, stance_seed, stance, total):
        count_prob = 0
        for i in range(len(seed)):
            if (seed[i] == word1):
                if (stance_seed[i] == stance):
                    count_prob = count_prob + 1
        return ((count_prob + 1))

    def prob_cond1(word1, word2, Features, total):
        return ((co_relation[(word1, word2)] + 1))

    raw = pd.read_csv(topic_Path+"//train_clean.csv",header=0)

    # Features Extraction
    porter = PorterStemmer()

    Stop_words = set(stopwords.words('english'))
    Features = raw[['Sentence']]
    Tweet = Features['Sentence'].copy()

    # Features['Sentence'] = Tweet.apply(sent_process)
    Features['Sentence'] = Tweet.apply(lambda x:sent_process(x, word_dict, norm_dict))
    Features['tokenized_sents'] = Features.apply(lambda row: (row['Sentence'].split()), axis=1)
    Features['pos_tag'] = Features.apply(lambda row: nltk.pos_tag(row['tokenized_sents'], tagset='universal'), axis=1)
    Features['stance'] = raw['Stance']
    length_Features = len(Features['Sentence'])

    co_relation = defaultdict(int)
    co_relation2 = []
    for i in range(length_Features):
        line = []
        for word, tag in Features['pos_tag'][i]:
            if (tag == 'NOUN' or tag == 'ADJ' or tag == 'VERB' or tag == 'ADV'):
                if (word not in Stop_words):
                    line.append(porter.stem(word))
        for i in range(len(line)):
            for j in range(i + 1, len(line)):
                co_relation[(line[i], line[j])] += 1
                co_relation[(line[j], line[i])] += 1
        co_relation2.append(line)

    #saving words after cleaning POS
    Features['co_relation'] = co_relation2


    len_co = []
    for i in range(length_Features):
        len_co.append(len(Features['co_relation'][i]))

    #gives number of words remained after preprocessing
    Features['len_nava'] = len_co

    nava = []
    for i in range(length_Features):
        for word, tag in Features['pos_tag'][i]:
            if (tag == 'NOUN' or tag == 'ADJ' or tag == 'VERB' or tag == 'ADV'):
                if (word not in Stop_words):
                    nava.append(word.lower())
    nava_stem = []
    for word in nava:
        nava_stem.append(porter.stem(word))
    uni_nava_stem = list(set(nava_stem))
    nava_stem = Counter(nava_stem)

    total = len(nava_stem)#total number of words
    length = len(uni_nava_stem)#total number of unique words

    print(total, length)

    seed = []
    non_seed = []
    seed_stance = []
    for i in range(len(Features)):
        for j in range(int(0.75 * Features['len_nava'][i])):
            seed.append(Features['co_relation'][i][j])
            seed_stance.append(Features['stance'][i])
        for j in range(int(0.75 * Features['len_nava'][i]), Features['len_nava'][i]):
            non_seed.append(Features['co_relation'][i][j])
    uni_seed = list(set(seed))
    uni_non_seed = list(set(non_seed))

    len_seed = len(seed)
    len_uni_seed = len(uni_seed)
    len_non_seed = len(non_seed)
    len_uni_non_seed = len(uni_non_seed)

    len_seed_label=[0]*len(labels_array)#this array counts len seed for each label. every i place represent len seed of label
    # len_seed_sup = 0
    # len_seed_opp = 0
    # len_seed_nut = 0
    for i in range(len(seed_stance)):
        for t in range(len(labels_array)):
            if(seed_stance[i] == labels_array[t]):
                len_seed_label[t]=len_seed_label[t]+1


    #returns for each word how many tims it appear, including number of corolations
    prob_word = []
    for word in uni_seed:
        prob_word.append(prob(word, nava_stem, total))

    prob_array_word = [] #this is array of arrays. each cell for label
    for i in range(len(labels_array)):
        prob_array_word.append([])

    for word in uni_seed:
        for i in range(len(labels_array)):
            prob_array_word[i].append(prob_cond(word, seed, seed_stance, labels_array[i], sum(len_seed_label)))


    prob_cond_word = {'word': list(uni_seed), 'prob_word': prob_word}
    Seed_lexicon = pd.DataFrame(data=prob_cond_word)

    # print(Seed_lexicon)

#this is array of arrays that have array of pmi for each label
    pmi_labels=[];
    for i in range(len(labels_array)):
        pmi_labels.append([])


    for i in range(len_uni_seed):
        for j in range(len(pmi_labels)):
            pmi_labels[j].append(pmi(prob_array_word[j][i], prob_word[i], len_seed_label[j], len_seed))



    for i in range(len(labels_array)):
        newName = "pmi_"+labels_array[i]
        Seed_lexicon[newName]=list(pmi_labels[i])


    stance = []
    for i in range(len_uni_seed):
        find_max_array = []
        for j in range(len(labels_array)):
            newName = "pmi_" + labels_array[j]
            find_max_array.append(Seed_lexicon[newName][i])
        label_index=find_max_array.index(max(find_max_array))
        stance.append(labels_array[label_index])


    Seed_lexicon['Stance'] = list(stance)


    score_non_seed_label = []
    seed_word_label=[]#this is array of arrays, for each label, contain array of words
    for i in range(len(labels_array)):
        seed_word_label.append([])
        score_non_seed_label.append([])



    for i in range(len_uni_seed):
        for t in range(len(seed_word_label)):
            if (Seed_lexicon['Stance'][i] == labels_array[t]):
                seed_word_label[t].append(Seed_lexicon['word'][i])




    print("COMPUTING...")

    for word in uni_non_seed:
        for j in range(len(seed_word_label)):
            list_ = []
            for i in range(len(seed_word_label[j])):
                l = pmi(prob_cond1(word, seed_word_label[j][i], Features, total), prob(word, nava_stem, total),
                        prob(seed_word_label[j][i], nava_stem, total), total)
                if (l < 0):
                    list_.append(1)
                else:
                    list_.append(l)

            score_non_seed_label[j].append(stats.gmean(list_))



    prob_cond_word = {'word': list(uni_non_seed)}

    NonSeed_lexicon = pd.DataFrame(data=prob_cond_word)

    # Tweet Vector Formation
    lex_word = []
    lex_word.extend(list(Seed_lexicon['word']))
    lex_word.extend(list(NonSeed_lexicon['word']))

    pmi_labels=[]
    for i in range(len(labels_array)):
        pmi_labels.append([])

    for j in range(len(pmi_labels)):
        newWord = "pmi_"+labels_array[j]
        pmi_labels[j].extend(list(Seed_lexicon[newWord]))
        pmi_labels[j].extend(list(score_non_seed_label[j]))


    Lexicon = dict()
    for i in range(len(lex_word)):
        Lexicon[lex_word[i]] = {}
        for j in range(len(labels_array)):
            newWord="pmi_"+labels_array[j]
            Lexicon[lex_word[i]][newWord]=pmi_labels[j][i]


    print("Lexicon formed")
    return Lexicon


def produce_features(labels_array,topic_path,Lexicon, word_dict, norm_dict):
    #train_features
    for l in ['train','test']:
        l1= l+"_clean"
        print("checking path:", l1)
        raw=pd.read_csv(topic_path+'\\{}.csv'.format(l1), header=0)

        Stop_words=set(stopwords.words('english'))
        Features=raw[['Sentence']]
        Tweet=Features['Sentence'].copy()

        Features['preprocessed_sentence']=Tweet.apply(lambda x:sent_process(x, word_dict, norm_dict))
        Features['tokenized_sents'] = Features.apply(lambda row: (row['preprocessed_sentence'].split()), axis=1)

        porter = PorterStemmer()
        start=time.time()


        columns_names_for_data = []
        columns_names_for_data.append('sentence')
        for i in range(len(labels_array)):
            newWord = "pmi_"+labels_array[i]
            columns_names_for_data.append(newWord)

        data = [columns_names_for_data]
        # len_lexicon_word=len(Lexicon)
        print("i received:"+str(len(Features['Sentence']))+" tweets")
        counter=0
        for i in range(len(Features['Sentence'])):
            sum_array_for_label=[0]*len(labels_array)

            total_lex=0
            temp = []
            for word in Features['tokenized_sents'][i]:
                #for j in range(len_lexicon_word):

                w = porter.stem(word)
                if w in Lexicon:
                    for b in range(len(sum_array_for_label)):
                        newWord = "pmi_"+labels_array[b]
                        sum_array_for_label[b]=sum_array_for_label[b]+Lexicon[w][newWord]

                    total_lex=total_lex+1

            if(total_lex==0):
                print("the problem is in:",Features['preprocessed_sentence'][i])
                print("sum are:", sum_array_for_label)
            array_for_data = []
            array_for_data.append(Features['Sentence'][i])
            for n in range(len(sum_array_for_label)):
                if(sum(sum_array_for_label)==0 or total_lex==0):
                    array_for_data.append(0.0)
                    print("i add 0 in index:",i)
                else:
                    array_for_data.append(sum_array_for_label[n]/total_lex)
            counter = counter + 1
            data.append(array_for_data)
            # data.append([Features['Tweet'][i],sum1/total_lex,sum2/total_lex,sum3/total_lex])
        print("i wrote to file:" + str(counter) + " rows")
        my_df = pd.DataFrame(data)
        my_df.to_csv(topic_path+'\\STA_feature_extraction_{}.csv'.format(l),header=False,index=False)


    end=time.time()
    print(end-start)

def run_STA_feature_extraction(labels_array):
    import os
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    word_dict, norm_dict = load_glove_embeddings_set(), create_normalise_dict()
    import os
    arr = os.listdir(BASE_DIR+"\\topics")
    for topic in arr:
        print("feature extraction - STA - for topic:", topic)
        produce_features(labels_array,BASE_DIR+"\\topics\\"+topic, build_lexicon(labels_array,BASE_DIR+"\\topics\\"+topic,word_dict, norm_dict), word_dict, norm_dict)

