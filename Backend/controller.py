from Backend.main_model import get_dataset_name, get_algorithmes_names, insert_dataset_to_db, get_topics_main, \
     get_one_stance, start_Specific_Model, get_model_result_main, get_req_details_main, get_models_desc_controller_main, \
     get_categories_dataset_main, get_5_sen_ds_main, get_labels_count_main, get_dataset_desc_controller_main, \
     get_topic_count_main, get_positive_negative_main


def get_dataset_name_controller():
     return get_dataset_name()



def get_algorithmes_names_controller():
     return get_algorithmes_names()

def add_dataset(csv_dataset):
     insert_dataset_to_db(csv_dataset)

def get_topics_controller():
     return get_topics_main()


def get_stance_controller(sentence,topic):
     return get_one_stance(sentence,topic)

def start_specific_model_controller(models, dataset_name, train_percent,df_extenal,based):
     return start_Specific_Model(models, dataset_name, train_percent,df_extenal,based)

def get_models_results_controller(model, dataset_name, train_percent):
     return get_model_result_main(model, dataset_name, train_percent)

def get_models_request_controller(id):
     return get_req_details_main(id)


def get_models_desc_controller(model):
     return get_models_desc_controller_main(model)


def get_categories_dataset_controller(dataset):
     return get_categories_dataset_main(dataset)


def get_5_sen_ds_controller(dataset):
     return get_5_sen_ds_main(dataset)


def get_labels_count_controller(dataset):
     return get_labels_count_main(dataset)


def get_topic_count_controller(dataset):
     return get_topic_count_main(dataset)




def get_dataset_desc_controller(datset):
     return get_dataset_desc_controller_main(datset)



def get_positive_negative_controller(dataset):
     return get_positive_negative_main(dataset)