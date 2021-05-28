
from Backend.TAN.pred import Pred,get_predict_per_stance
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



import pandas as pd
def test_Pred():


    test_labels,all_topic_labels=Pred(df_train,df_test,labels,num_of_labels)
    is1=isinstance(test_labels,list)
    is2=isinstance(all_topic_labels,list)


def get_predict_per_stance_test():
    sentence="I love hilary clinton"
    claim="Hilary Clinton"
    label=get_predict_per_stance(sentence, claim, None)
    isinstance(label,label in labels)

