

from Backend.TRANSFORMER.measure import predict
from Backend.TRANSFORMER.mymodels import getModelWithType
from Backend.TRANSFORMER.measure import predict
from Backend.TRANSFORMER.utils import loadTokenizer
from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels
from Backend.DB.DBManager import DataBase
import numpy as np


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

def test_predict():
    test_labels_return,result_return=predict(df_test,dataset_name,labels)
    is1=isinstance(test_labels_return,list)
    is2=isinstance(result_return,list)

    assert (is1, True)
    assert (is2, True)


