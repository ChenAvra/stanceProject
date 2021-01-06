import os

from gensim.models import KeyedVectors
import  numpy as np
from .early_stopping_training import run_model, pred_one_stance
from ..DB.DBManager import DataBase

embedding_matrix_per_claim = {}
len_ensemble_models_all={}
def Pred(df_train, df_test, labels, num_of_labels):
    all_topic_labels=[]
    all_topic_y_test=[]

    for claim in (df_train.Claim.unique()):
        df_train_per_claim=df_train.loc[df_train['Claim'] == claim]
        df_test_per_claim=df_test.loc[df_test['Claim'] == claim]

        labels_pred, y_test, len_ensemble_model, labels, embedding_matrix=run_model(df_train_per_claim, df_test_per_claim, labels, num_of_labels,claim)
        embedding_matrix_per_claim.update({claim:embedding_matrix})
        len_ensemble_models_all.update({claim:len_ensemble_model})
        file_name = str(claim).replace(' ', "")
        file_name1 = str(file_name).replace('-', "")
        file_name2 = str(file_name1).replace('?', "")

        PROJECT_ROOT = os.path.abspath(__file__)
        BASE_DIR = os.path.dirname(PROJECT_ROOT)
        WORD2VEC_MODEL = BASE_DIR + '\\'+file_name2+'.txt'
        with open(WORD2VEC_MODEL, 'wb') as f:
            np.save(f, embedding_matrix)
        # embedding_matrix.save("word2vec."+str(claim))
        all_topic_labels.extend(labels_pred)
        all_topic_y_test.extend(y_test)

    return all_topic_labels, all_topic_y_test

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
    with open(WORD2VEC_MODEL, 'rb') as f:
        embedding_matrix_per_claim = np.load(f)


    dataset_id = dataset_names_dict[dataset]
    db = DataBase()
    df = db.get_dataset(dataset_id)
    labels=df.Stance.unique()

    stance=pred_one_stance(labels,embedding_matrix_per_claim,sentence,claim,stance)
    return stance


# print(get_predict_per_stance(2,"I think it's bad",'Are E-Cigarettes safe?'))