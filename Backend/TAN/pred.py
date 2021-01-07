import os
import pickle

from gensim.models import KeyedVectors
import  numpy as np
from .early_stopping_training import run_model, pred_one_stance
from ..DB.DBManager import DataBase

embedding_matrix_per_claim = {}
len_ensemble_models_all={}
def Pred(df_train, df_test, labels, num_of_labels):
    all_topic_labels=[]
    all_topic_y_test=[]

    claim="news are important to our life"

    stances = {}

    for i in range(len(labels)):
        stances.update({labels[i]: i})

    # list=['Feminist Movement','Hillary Clinton','Legalization of Abortion']
    # for claim in (df_train.Claim.unique()):
    # for claim in list:
    #     df_train_per_claim=df_train.loc[df_train['Claim'] == claim]
    #     df_test_per_claim=df_test.loc[df_test['Claim'] == claim]

    # labels_pred, y_test, len_ensemble_model, labels, embedding_matrix,word_ind=run_model(df_train_per_claim, df_test_per_claim, labels, num_of_labels,claim)
    labels_pred, y_test, len_ensemble_model, labels, embedding_matrix, word_ind = run_model(df_train,
                                                                                            df_test, labels,
                                                                                            num_of_labels, claim)

    embedding_matrix_per_claim.update({claim: embedding_matrix})
    len_ensemble_models_all.update({claim: len_ensemble_model})
    file_name = str(claim).replace(' ', "")
    file_name1 = str(file_name).replace('-', "")
    file_name2 = str(file_name1).replace('?', "")

    # write word ind to dictionary file
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    word_ind_file = BASE_DIR + '\\' + file_name2 + '.pkl'

    a_file = open(word_ind_file, "wb")
    pickle.dump(word_ind, a_file)
    a_file.close()
    # save word embeddong
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    WORD2VEC_MODEL = BASE_DIR + '\\' + file_name2 + '.txt'
    with open(WORD2VEC_MODEL, 'wb') as f:
        np.save(f, embedding_matrix)
    # embedding_matrix.save("word2vec."+str(claim))
    all_topic_labels.extend(labels_pred)
    all_topic_y_test.extend(y_test)

    test_labels=[]
    for label in all_topic_y_test:
        for i in range(len(labels)):
            sivug = labels[i]
            st = stances[sivug]
            if st == label:
                test_labels.append(sivug)


    return  test_labels,all_topic_labels

def get_predict_per_stance(sentence,claim,stance):
    dataset=""

    listMPCHI=['Are E-Cigarettes safe?', 'Does Sunlight exposure lead to skin cancer?',
                       'Does Vitamin C prevent common cold?', 'Should women take HRT post-menopause?',
                       'Does MMR Vaccine lead to autism in children?' ]
    listSemEval2016=['Atheism', 'Hillary Clinton',
                       'Legalization of Abortion', 'Climate Change is a Real Concern',
                       'Feminist Movement']

    if claim in listMPCHI:
        dataset ='MPCHI'
    else:
        dataset='semEval2016'
    dataset_names_dict = {
        "semEval2016": 1,
        "semEval2017": 2,
        "FNC": 3,
        "MPCHI": 4,
        "EmergentLite": 5,
    }

    file_name = str(claim).replace(' ', "")
    file_name1 = str(file_name).replace('-', "")
    file_name2 = str(file_name1).replace('?', "")

    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    WORD2VEC_MODEL = BASE_DIR + '\\' + file_name2 + '.txt'
    # model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=False)
    embedding_matrix_per_claim=[]
    #load embedding matrix
    with open(WORD2VEC_MODEL, 'rb') as f:
        embedding_matrix_per_claim = np.load(f)

    #load word_ind
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    word_ind_file = BASE_DIR + '\\' + file_name2 + '.pkl'
    a_file = open(word_ind_file, "rb")
    word_ind_load = pickle.load(a_file)
    a_file.close()

    dataset_id = dataset_names_dict[dataset]
    db = DataBase()
    df = db.get_dataset(dataset_id)
    labels=df.Stance.unique()

    stance=pred_one_stance(labels,embedding_matrix_per_claim,sentence,claim,stance,word_ind_load)
    return stance


# print(get_predict_per_stance(2,"I think it's bad",'Are E-Cigarettes safe?'))