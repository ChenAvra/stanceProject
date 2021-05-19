from Backend.main_model import *


# def insert_dataset_to_db(path_dataset):
#     db = DataBase()
#     num = db.index_dataset
#     db.insert_external_dataset(path_dataset,num)
#     db.index_dataset = db.index_dataset+1

def get_topics_main():
    db = DataBase()
    return db.get_topics_db()


def get_model_result_main(model, dataset_name, train_percent):
    db = DataBase()
    return db.get_record_from_result(model, dataset_name, train_percent)

def get_req_details_main(id):
    db = DataBase()
    return db.get_records_from_request_by_id(id)


def get_models_desc_controller_main(model):
    db = DataBase()
    return db.get_model_desc_db(model)


def get_categories_dataset_main(dataset):
    db = DataBase()

    dataset_id = dataset_names_dict[dataset]

    df = db.get_dataset(dataset_id)
    labels = get_unique_labels(df)
    return labels


def get_5_sen_ds_main(dataset):
    db = DataBase()

    dataset_id = dataset_names_dict[dataset]

    df = db.get_dataset(dataset_id)
    df = df.head(5)
    return df



def get_labels_count_main(dataset):
    db = DataBase()

    dataset_id = dataset_names_dict[dataset]

    df = db.get_dataset(dataset_id)
    labels = get_unique_labels(df)
    num_labels = []
    labels_arr = []
    for label in labels:
        num = (df[df['Stance'] == label]).shape[0]
        num_labels.append(num)
        labels_arr.append(label + "(" + str(num) + ")")

    return num_labels,labels_arr



def get_topic_count_main(dataset):
    db = DataBase()
    name_topics=[]
    topics_arr=[]

    dataset_id = dataset_names_dict[dataset]
    df = db.get_dataset(dataset_id)

    topics = df.Claim.unique()

    for topic in topics:
        num = (df[df['Claim'] == topic]).shape[0]
        name_topics.append(topic)
        topics_arr.append(num)

    return name_topics,topics_arr


def get_dataset_desc_controller_main(dataset):
    db = DataBase()
    return db.get_dataset_desc_db(dataset)


def get_num_of_records_controller(dataset):
    db = DataBase()
    dataset_id = dataset_names_dict[dataset]
    num = db.get_dataset(dataset_id).shape[0]
    return num


def get_positive_negative_main(dataset):
    from afinn import Afinn
    afinn = Afinn()
    count_pos = 0
    count_neg = 0
    count_net = 0
    # afinn.score('This is utterly excellent!')
    db = DataBase()

    dataset_id = dataset_names_dict[dataset]
    df = db.get_dataset(dataset_id)

    for i in range(df.shape[0]):
        sen = df.iloc[i]['Sentence']
        score = afinn.score(sen)
        if score > 0:
            count_pos = count_pos + 1
        elif score < 0:
            count_neg = count_neg + 1
        else:
            count_net = count_net + 1

    return [count_pos,count_neg,count_net]

