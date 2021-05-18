from Backend.Allada_Nandakumar.model import *
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

def test_text_cleaner1():
    train_body = [text_cleaner(body) for body in df_train['Sentence']]
    assert len(train_body) == len(df_train)

def test_text_cleaner2():
    train_body = [text_cleaner(body) for body in df_train['Sentence']]
    for i in range(len(train_body)):
        assert " the " not in train_body[i]