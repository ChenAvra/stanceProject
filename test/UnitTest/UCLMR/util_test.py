from Backend.UCLMR.util import FNCData, pipeline_train, pipeline_test
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

def test_FNCData_constructor1():
    raw_train = FNCData(df_train)
    assert isinstance(raw_train, object)

def test_FNCData_constructor2():
    raw_train = FNCData(df_train)
    assert isinstance(raw_train.bodies, dict)

def test_FNCData_constructor3():
    raw_train = FNCData(df_train)
    assert isinstance(raw_train.instances, list)

def test_FNCData_constructor4():
    raw_train = FNCData(df_train)
    assert isinstance(raw_train.heads, dict)

def test_FNCData_constructor5():
    raw_train = FNCData(df_train)
    assert len(raw_train.heads) == num_of_topics

def test_dfInstance1():
    raw_train = FNCData(df_train)
    instances = raw_train.dfInstance(df_train)
    assert isinstance(instances, list)

def test_dfInstance2():
    raw_train = FNCData(df_train)
    instances = raw_train.dfInstance(df_train)
    assert isinstance(instances[0], dict)

def test_dfInstance3():
    raw_train = FNCData(df_train)
    instances = raw_train.dfInstance(df_train)
    assert len(instances[0]) == 4

def test_dfBodies1():
    raw_train = FNCData(df_train)
    bodies = raw_train.dfBodies(df_train)
    assert isinstance(bodies, list)

def test_dfBodies2():
    raw_train = FNCData(df_train)
    bodies = raw_train.dfBodies(df_train)
    assert isinstance(bodies[0], dict)

def test_dfBodies3():
    raw_train = FNCData(df_train)
    bodies = raw_train.dfBodies(df_train)
    assert len(bodies[0]) == 4

def test_pipeline_train1():
    raw_train = FNCData(df_train)
    raw_test = FNCData(df_test)
    lim_unigram = 5000

    label_ref = {}
    counter = 0
    for t in labels:
        label_ref[t] = counter
        counter += 1

    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, label_ref, lim_unigram=lim_unigram)

    assert len(train_set) == len(raw_train.bodies)


def test_pipeline_train2():
    raw_train = FNCData(df_train)
    raw_test = FNCData(df_test)
    lim_unigram = 5000

    label_ref = {}
    counter = 0
    for t in labels:
        label_ref[t] = counter
        counter += 1

    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, label_ref, lim_unigram=lim_unigram)

    assert len(train_stances) == len(raw_train.bodies)

def test_pipeline_test1():
    raw_train = FNCData(df_train)
    raw_test = FNCData(df_test)
    lim_unigram = 5000

    label_ref = {}
    counter = 0
    for t in labels:
        label_ref[t] = counter
        counter += 1

    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, label_ref, lim_unigram=lim_unigram)

    test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    assert isinstance(test_set, list)


