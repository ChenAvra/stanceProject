from Backend.DB.DBManager import DataBase
from Backend.LIU.Feature_Extract import FeatureExtract
from Backend.LIU.main import main, runLIU
from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels

db = DataBase()
dataset_name = "semEval2016"
dataset_id = dataset_names_dict[dataset_name]
df = db.get_dataset(dataset_id)
df_train, df_test = split_data_topic_based(df, 80)
df_train_records = df_train.shape[0]
df_test_records = df_test.shape[0]
type = 'topic'
labels = get_unique_labels(df)
num_of_labels = len(labels)


def test_data_to_tf1():
    tf = FeatureExtract().data_to_tf(df_train)
    assert len(tf) == len(df_train)


def test_data_to_tf2():
    tf = FeatureExtract().data_to_tf(df_train)
    assert len(tf[0]) == 5000

def test_tfidf_lookup():
    lookUp = FeatureExtract().tfidf_lookup(df_train)
    assert type(lookUp) is dict

def test_load_word_embedding():
    dictionary = FeatureExtract().__load_word_embedding()
    assert type(dictionary) is dict