from Backend.DB.DBManager import DataBase
from Backend.LIU.Feature_Extract import FeatureExtract
from Backend.LIU.main import main, runLIU, OVER_SAMPLING
from Backend.SEN.SEN import get_legal_directort_name, split_to_topic_folders, create_normalise_dict
from Backend.SEN.preprocessing import clean_str, load_glove_embeddings, split
from Backend.SEN.te_f import sent_process, isnan, sentence2sequence, training_model, score_setup
from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels
from Backend.LIU.DataSet import DataSet
import numpy as np
import os

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

def test_sent_proce():
    clean_word = sent_process("THIS IS AN APPLE")
    assert "APPLE" == clean_word

def test_isnan():
    is_nan = isnan("5")
    assert False == is_nan


def test_score_setup():
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    import csv
    with open(BASE_DIR+"\\snli_1.0_dev.txt", "r") as data:
        train = csv.DictReader(data, delimiter='\t')
        for row in train:
            number = score_setup(row)
            break
    assert isnan(number[0])==False