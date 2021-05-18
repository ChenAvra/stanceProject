from Backend.main_model import *

db = DataBase()
dataset_name = "semEval2016"
dataset_id = dataset_names_dict[dataset_name]
df = db.get_dataset(dataset_id)

def test_get_dataset_name():
    dataset_names = get_dataset_name()
    assert len(dataset_names) == 10

def test_get_algorithmes_names():
    algo_names = get_algorithmes_names()
    assert len(algo_names) == 6

def test_split_data_topic_based1():
    df_train, df_test = split_data_topic_based(df, 0.8)
    assert len(df_train) + len(df_test) == len(df)

def test_split_data_topic_based2():
    df_train, df_test = split_data_topic_based(df, 0.8)
    assert round(len(df_train) / len(df), 2) == 0.8

def test_get_unique_labels():
    labels = get_unique_labels(df)
    assert len(labels) == 3

def test_get_one_stance():
    stance = get_one_stance("I think she is a nice woman", 'Hillary Clinton')
    print(stance)
