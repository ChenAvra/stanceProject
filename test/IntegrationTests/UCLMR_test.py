from Backend.UCLMR.pred import *
from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels
from Backend.DB.DBManager import DataBase

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
topics = df.Claim.unique()
num_of_topics = len(topics)

def test_Pred1():
    y_test, y_pred = Pred(df_train, df_test, labels, num_of_labels)
    assert len(y_test) == len(y_pred)

def test_Pred2():
    y_test, y_pred = Pred(df_train, df_test, labels, num_of_labels)
    assert len(y_pred) == len(df_test)

def test_Pred3():
    y_test, y_pred = Pred(df_train, df_test, labels, num_of_labels)
    y_pred_np = np.array(y_pred)
    assert len(np.unique(y_pred_np)) == num_of_labels