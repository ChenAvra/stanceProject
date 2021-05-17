# from Backend.DB.DBManager import DataBase
# from Backend.LIU.main import main, runLIU
# from Backend.main_model import dataset_names_dict, split_data_topic_based, get_unique_labels
#
# db = DataBase()
# dataset_name = "semEval2016"
# dataset_id = dataset_names_dict[dataset_name]
# df = db.get_dataset(dataset_id)
# df_train, df_test = split_data_topic_based(df, 80)
# df_train_records = df_train.shape[0]
# df_test_records = df_test.shape[0]
# type = 'topic'
# labels = get_unique_labels(df)
# num_of_labels = len(labels)
#
#
# def test_main1():
#     # y_test, y_pred = runLIU(df_train, df_test, labels, num_of_labels)
#     y_test, y_pred = main().classify(df_train, df_test, labels, num_of_labels)
#     assert len(y_pred) == len(y_test)
#
# def test_main2():
#     # y_test, y_pred = runLIU(df_train, df_test, labels, num_of_labels)
#     y_test, y_pred = main().classify(df_train, df_test, labels, num_of_labels)
#     assert len(y_pred) == len(df_test)