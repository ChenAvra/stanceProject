

from Backend.TRANSFORMER.textpreprocessing import preprocessDF
from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels
from Backend.DB.DBManager import DataBase
import numpy as np
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
def test_preprocess():
    df_merged, stances=preprocessDF(df_train,labels)
    isNumber=np.issubdtype(df_merged['Stance'].dtype, np.number)
    s1=isinstance(df_merged,pd.DataFrame)
    assert (isNumber, True)



