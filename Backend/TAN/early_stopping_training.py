import sys
import csv
import copy
import numpy as np
import re
import itertools
from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from .utils import *

# from Preprocessing.preprocessing import *

# torch.device('cpu')
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from .networks import *
import pickle
from datetime import datetime
import random
from statistics import mode
import copy
import os
import pandas as pd
import matplotlib.pyplot as plt

D = None

random.seed(42)

# if len(sys.argv) !=3:
#     print("Usage :- python early_stopping_training_test.py <dataset name> <attention vairant>")
#     sys.exit(-1)

version = "tan"
# dataset = "EC"
import torch.nn.functional as F
def f_score(table):
    return "%.2f" % (100*table[0][0]/(table[0][1]+table[0][2]) + 100*table[1][0]/(table[1][1]+table[1][2]))


def train_bagging_tan_CV(stances,x_train, y_train, x_test, y_test, vector_target,labels,device,embedding_matrix,claim,version="tan-",n_epochs=1,batch_size=50,l2=0,dropout = 0.5,n_folds=2):

    NUM_EPOCHS = n_epochs
    loss_fn = nn.NLLLoss()
    n_models = 1
    print("\n\n starting cross validation \n\n")
    print("class : ",claim, " :-")

    score = 0

    fold_sz = len(x_train)//n_folds
    foldwise_val_scores = []
    ensemble_models = []
    print("dataset size :- ",len(x_train))
    for fold_no in range(n_folds):
        print("Fold number {}".format(fold_no+1))
        best_val_score = 0
        model = LSTM_TAN(version,300,100,len(embedding_matrix),len(labels),embedding_matrix,dropout=dropout).to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=0.0005,weight_decay = l2)
        if fold_no == n_folds-1:
            ul = len(x_train)
        else:
            ul = (fold_no+1)*fold_sz
        print("ll : {}, ul : {}".format(fold_no*fold_sz,ul))
        best_ensemble = []
        best_score = 0
        temp_ensemble = []
        temp_score = 0
        for _ in range(NUM_EPOCHS):
            ep_loss = 0
            target = torch.tensor(vector_target,dtype=torch.long).to(device)
            optimizer.zero_grad()

            #training

            model.train()
            loss = 0




            for i in range(fold_no*fold_sz):
                model.hidden = model.init_hidden()

                x = torch.tensor(np.array(x_train[i]),dtype=torch.long).to(device)
                y = torch.tensor([y_train[i]],dtype=torch.long).to(device)

                preds = model(x,target,verbose=False)

                x_ = loss_fn(preds,y)
                loss += x_
                ep_loss += x_
                if (i+1) % batch_size == 0:
                    loss.backward()
                    loss = 0
                    optimizer.step()
                    optimizer.zero_grad()

            for i in range(ul,len(x_train)):
                model.hidden = model.init_hidden()

                x = torch.tensor(np.array(x_train[i]),dtype=torch.long).to(device)
                y = torch.tensor([y_train[i]],dtype=torch.long).to(device)

                preds = model(x,target,verbose=False)

                x_ = loss_fn(preds,y)
                loss += x_
                ep_loss += x_
                if (i+1) % batch_size == 0:
                    loss.backward()
                    loss = 0
                    optimizer.step()
                    optimizer.zero_grad()

            optimizer.step()
            optimizer.zero_grad()
            print("finish")
            #validation
            corr = 0
            with torch.no_grad():
                conf_matrix = np.zeros((2,len(labels)))
                labels_validation=[]
                y_labels_test=[]
                for j in range(fold_sz*fold_no,ul):
                    x = torch.tensor(np.array(x_train[j]),dtype=torch.long).to(device)
                    y = torch.tensor([y_train[j]],dtype=torch.long).to(device)
                    model.eval()
                    preds = model(x,target,verbose=False)
                    label = np.argmax(preds.cpu().numpy(),axis=1)[0]
                    labels_validation.append(label)
                    y_labels_test.append(y_train[j])
                    # accuracy_validation=metrics.accuracy_score(labels_validation,y_labels_test)
                    # if label == y_train[j]:
                    #     corr+=1
                    #     if label <=1:
                    #         conf_matrix[label][0]+=1
                    # if y_train[j] <=1:
                    #     conf_matrix[y_train[j]][2]+=1
                    # if label <=1:
                    #     conf_matrix[label][1]+=1
                    ep_loss+=loss_fn(preds,y)
            accuracy_validation = metrics.accuracy_score(labels_validation, y_labels_test)
            # val_f_score = float(f_score(conf_matrix))
            val_f_score=accuracy_validation

            if val_f_score > best_val_score:
               best_val_score = val_f_score
               best_model = copy.deepcopy(model)


            if _%10 ==0 and _ != 0:
                print("current last 10- score ",temp_score*1.0/10)
                if temp_score > best_score:
                    best_score = temp_score
                    best_ensemble = temp_ensemble
                    print("this is current best score now")
                temp_ensemble = []
                temp_score = 0

            temp_ensemble.append(copy.deepcopy(model))
            temp_score += val_f_score



            print("epoch number {} , val_f_score {}".\
            # format(_+1,f_score(conf_matrix)))
            format(_+1,accuracy_validation))


        print("current last 10- score ",temp_score*1.0/10)
        if temp_score > best_score:
            best_score = temp_score
            best_ensemble = temp_ensemble
            print("this is current best score now")
        # if not best_ensemble:
        #     ensemble_models.extend(temp_ensemble)
        ensemble_models.extend(best_ensemble)
    labels_pred=[]
    labels_prob=[]
    with torch.no_grad():
        # conf_matrix = np.zeros((2,len(labels)))
        for j in range(len(x_test)):
            x = torch.tensor(np.array(x_test[j]),dtype=torch.long).to(device)
            y = torch.tensor([y_test[j]],dtype=torch.long).to(device)
            all_preds = []
            for model in ensemble_models:
                model.eval()
                all_preds.append(np.argmax(model(x,target).cpu().numpy(),axis=1)[0])
                prob = F.softmax(model(x,target), dim=1)
                labels_prob.append(prob)
            cnts = np.zeros(len(labels))
            for prediction in all_preds:
                cnts[prediction]+=1
            label = np.argmax(cnts)
            labels_pred.append(label)
            # if label == y_test[j]:
            #     corr+=1
            #     if label <=1:
            #         conf_matrix[label][0]+=1
            # if y_test[j] <=1:
            #     conf_matrix[y_test[j]][2]+=1
            # if label <=1:
            #     conf_matrix[label][1]+=1
            # ep_loss+=loss_fn(preds,y)

    conf_matrix = confusion_matrix(labels_pred, y_test)
    print(conf_matrix)

    score = metrics.accuracy_score(y_test, labels_pred)
    print(score)
    # print("test_f_score {}".format(f_score(conf_matrix)))
    models_to_save={}
    i=0
    for model in ensemble_models:
         models_to_save.update({i : model.state_dict()})
         i=i+1
    # models_to_save.update({'optimizer': optimizer.state_dict()})
    name_file=str(claim).replace(" ",'')
    name_file=str(name_file).replace("-","")
    name_file=str(name_file).replace("?","")

    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    model_path = BASE_DIR + '\\save' + name_file + '.pt'

    torch.save(models_to_save,model_path)

    labels_pred_strings=list()
    #change labels_pred to strings
    for label in labels_pred:
        for i in range(len(labels)):
            sivug=labels[i]
            st= stances[sivug]
            if st == label:
                labels_pred_strings.append(sivug)

    return conf_matrix,labels_pred_strings,score,model,len(ensemble_models)



# dataset ="EC"
# #add parameter
# dataset ="HRT"



# labels=[]
# df=pd.read_csv('C:\\Users\\Chen\\Desktop\\HRT.csv')

# topic_string="HRT"
# col=df['Stance']
# labels=df.Stance.unique()
# topics=df.Claim.unique()
# fin_matrix = np.zeros((2,len(labels)))

# stances = {}

# for i in range(len(labels)):
#     stances.update({labels[i]: i})
# embedding_matrix=[]

# x_train=[], y_train=[], x_test=[], y_tes=[], vector_targe=[]
def run_model(df_train, df_test, labels, num_of_labels,claim):
    version='tan'
    topic_string=claim

    stances, word2emb, word_ind, ind_word, embedding_matrix, device, \
    x_train, y_train, x_test, y_test, vector_target, train_tweets, test_tweets = load_dataset(topic_string,df_train,df_test,labels,claim, dev="cpu")


    combined = list(zip(x_train, y_train))
    random.shuffle(combined)
    x_train[:], y_train[:] = zip(*combined)


    fin_matrix,labels_pred,score,model,len_ensemble_model= train_bagging_tan_CV(stances,x_train, y_train, x_test, y_test, vector_target,labels,device,embedding_matrix,claim,version=version,n_epochs=20,batch_size=50,l2=1.0,dropout = 0.5,n_folds=3)

    return labels_pred,y_test,len_ensemble_model,labels,embedding_matrix,word_ind



def pred_one_stance(labels,embedding_matrix,sentence,claim,stance,word_ind_load):
    models_after_load={}
    stances = {}
    topic_string=''
    device = torch.device('cpu')

    for i in range(len(labels)):
        stances.update({labels[i]: i})

    file_name = str(claim).replace(' ', "")
    file_name1 = str(file_name).replace('-', "")
    file_name2 = str(file_name1).replace('?', "")

    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    path = BASE_DIR + '\\save' + file_name2 + '.pt'
    checkpoint = torch.load(path)

    len_ensemble_model=len(checkpoint)
    for i in range(len_ensemble_model):
        modelA = LSTM_TAN(version, 300, 100, len(embedding_matrix), len(labels), embedding_matrix, dropout=0.6).to('cpu')
        modelA.load_state_dict(checkpoint[i])
        models_after_load.update({i:modelA})

    # optimizerB = TheOptimizerBClass(*args, **kwargs)
    Y_test=['AGAINST']
    X_test=[sentence]
    df2 = pd.DataFrame(list(zip(Y_test, X_test)),
                   columns =['Stance', 'Sentence'])
    x_after_proc, vector_target_after_proc  = pre_proce_one_stance(word_ind_load,topic_string,df2,labels,claim,dev = "cpu")
    with torch.no_grad():
        # conf_matrix = np.zeros((2, len(labels)))
        for j in range(len(x_after_proc)):
            target = torch.tensor(vector_target_after_proc, dtype=torch.long).to(device)

            x = torch.tensor(np.array(x_after_proc[j]), dtype=torch.long).to(device)
            # y = torch.tensor([y_test[j]], dtype=torch.long).to(device)
            all_preds = []
            for index in models_after_load:
                model= models_after_load[index]
                model.eval()
                all_preds.append(np.argmax(model(x, target).cpu().numpy(), axis=1)[0])
            cnts = np.zeros( len(labels))
            for prediction in all_preds:
                cnts[prediction] += 1
            label = np.argmax(cnts)


    for i in range(len(labels)):
        if stances[labels[i]]==label:
            return (labels[i])

