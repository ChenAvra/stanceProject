
from Backend.TRANSFORMER.pred import Pred,Pred_one_sentence
from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels
from Backend.DB.DBManager import DataBase


db = DataBase()
dataset_name = "semEval2016"
dataset_id = dataset_names_dict[dataset_name]
df = db.get_dataset(dataset_id)
df_train, df_test = split_data_topic_based(df, 0.8)
df_train_records = df_train.shape[0]
df_test_records = df_test.shape[0]
type = 'topic'
labels = get_unique_labels(df)
num_of_labels = len(labels)
topics = df.Claim.unique()
num_of_topics = len(topics)

MODEL_NAME = "model_1"
MODEL_TYPE = "TRANSFORMER"

import pandas as pd
def test_Pred_one_sentence():
    claim="Hilary Clinton"
    sentence="I love hialry clinton"
    d = {'Claim': [claim], 'Sentence': [sentence], 'Stance': ['AGAINST']}
    df = pd.DataFrame(data=d)
    label=Pred_one_sentence(None,df,labels,num_of_labels,dataset_name)
    if label not in labels:
        isLabel=False
    else:
        isLabel=True

    assert (isLabel, True)
