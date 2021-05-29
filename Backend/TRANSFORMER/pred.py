import os



import  numpy as np
from .train import run_model
from .measure import predict
from ..DB.DBManager import DataBase

MODEL_NAME = "model_1"
MODEL_TYPE = "TRANSFORMER"

def Pred(df_train, df_test, labels, num_of_labels,dataset_name,train_percent):
    df_test_copy=df_test.copy(deep=True)
    PROJECT_ROOT = os.path.abspath('__file__')
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    checkpoint_path = BASE_DIR+"\\TRANSFORMER\\checkpoints\\"+MODEL_TYPE+"_"+MODEL_NAME+dataset_name+train_percent+'weights.hdf5'
    # checkpoint_path = BASE_DIR+"\\TRANSFORMER\\checkpoints\\"+MODEL_TYPE+"_"+MODEL_NAME+dataset_name+'weights.hdf5'

    if(not os.path.exists(checkpoint_path)):
        run_model(df_train, df_test, labels, num_of_labels,dataset_name,train_percent)
    test_labels, preds ,probs= predict(df_test_copy, dataset_name,labels,None,train_percent)

    return test_labels, preds,probs

def Pred_one_sentence(df_train, df_test, labels, num_of_labels,dataset_name,model_TRANSFORMER,train):
    df_test_copy=df_test.copy(deep=True)
    PROJECT_ROOT = os.path.abspath('__file__')
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    checkpoint_path = BASE_DIR+"\\TRANSFORMER\\checkpoints\\"+MODEL_TYPE+"_"+MODEL_NAME+dataset_name+'weights.hdf5'
    if(not os.path.exists(checkpoint_path)):
        run_model(df_train, df_test, labels, num_of_labels,dataset_name)
    test_labels, preds,all_probs = predict(df_test_copy, dataset_name,labels,model_TRANSFORMER,train,True)
    for i in range(len(labels)):
        if labels[i]==preds[0]:
            return labels[i]
