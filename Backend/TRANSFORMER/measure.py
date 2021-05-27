from imblearn import keras

from .dataset import readDataset,readTestDataset
from .textpreprocessing import preprocessDF, getBalancedData
from .utils import initTokenizer,loadTokenizer,dataPrep
from .mymodels import getModelWithType
import tensorflow as tf
import os
import numpy

## BEFORE RUNNING THIS FILE MAKE SURE THAT CONFIGURATION MATCHES FOR THE MODEL YOU WANT TO EVALUATE.

TEST_ON_COMP = False
## assign False if validation set should be extracted from training set itself
DATA_SPLIT = 0.2 
## Not applicable if TEST_ON_COMP is True
BINARY_CLASSIFICATION = False
## If True classes will be seperated for unrelated and related only
OVERSAMPLING_STANCEWISE = True
## Toggle for switching off or on oversampling for data according stance distribution
MODEL_NAME = "model_1"
NEW_TOKENIZER = True
## Toggle for loading previously created tokenizer with model name
MODEL_TYPE = "TRANSFORMER"
## TRANSFORMER or CNN

MAX_LENGTH_ARTICLE = 1200
MAX_LENGTH_HEADLINE = 40
TRAIN_EMBED = False
LOAD_PREV = False

# print("reading datasets ... ")
# main_set = readTestDataset("./dataset/")
# main_set = preprocessDF(main_set,binaryclass=BINARY_CLASSIFICATION)

# if(OVERSAMPLING_STANCEWISE):
#     main_set = getBalancedData(main_set,binaryclass=BINARY_CLASSIFICATION)
# tf.keras.backend.clear_session()


def predict(main_set,dataset_name,labels,model):
    print("MAIN SET balanced.")
    main_set,stances = preprocessDF(main_set, labels, binaryclass=BINARY_CLASSIFICATION)

    tokenizer = loadTokenizer(MODEL_NAME)

    print("DATA PREP for TEST set")
    testap,testhp,test_labels = dataPrep(main_set,tokenizer,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE)
    print("Padded Inputs with Labels TEST READY.")
    #
    # PROJECT_ROOT = os.path.abspath('__file__')
    # BASE_DIR = os.path.dirname(PROJECT_ROOT)
    # num_labels=len(labels)
    # checkpoint_path = BASE_DIR+"\\TRANSFORMER\\checkpoints\\"+MODEL_TYPE+"_"+MODEL_NAME+dataset_name+"weights.hdf5"
    # # checkpoint_path = "./checkpoints/"+MODEL_TYPE+"_"+MODEL_NAME+"/weights.hdf5"
    # model_ = getModelWithType(MODEL_TYPE,BINARY_CLASSIFICATION,MAX_LENGTH_ARTICLE,MAX_LENGTH_HEADLINE,TRAIN_EMBED,tokenizer,num_labels)
    # model_.load_weights(checkpoint_path)
    # print(model_.summary())
    # tf.keras.backend.clear_session()
    result = model.predict([testap,testhp],batch_size=100)

    print(result)
    result=numpy.argmax(result, axis=1)
    test_labels=numpy.argmax(test_labels, axis=1)

    test_labels_return = []
    for label in test_labels:
        for i in range(len(labels)):
            sivug = labels[i]
            st = stances[sivug]
            num=int(label)
            if st==num:
                test_labels_return.append(sivug)
                break



    result_return = []
    for label in result:
        for i in range(len(labels)):
            sivug = labels[i]
            st = stances[sivug]
            if st == label:
                result_return.append(sivug)



    return test_labels_return,result_return




