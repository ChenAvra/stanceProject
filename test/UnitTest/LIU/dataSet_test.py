from Backend.DB.DBManager import DataBase
from Backend.LIU.Feature_Extract import FeatureExtract
from Backend.LIU.main import main, runLIU, OVER_SAMPLING
from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels
from Backend.LIU.DataSet import DataSet
import numpy as np

db = DataBase()
dataset_name = "semEval2016"
dataset_id = dataset_names_dict[dataset_name]
df = db.get_dataset(dataset_id)
df_train, df_test = split_data_topic_based(df, 0.80)
df_train_records = df_train.shape[0]
df_test_records = df_test.shape[0]
type = 'topic'
labels = get_unique_labels(df).tolist()
num_of_labels = len(labels)
ds = DataSet(preprocess=False, labelsFromDB=labels, train_df=df_train, test_df=df_test)
train_all = ds.get_train()

def test_getTrain():
    test = ds.get_test()
    assert len(test) == len(df_test)
