import os
import sklearn
import sklearn.model_selection as model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, label_binarize, LabelEncoder
from datetime import time, datetime

from Backend.DB.DBManager import *
from Backend.LIU.runLIU import LIU
from Backend.TRANSFORMER.runTRANSFORMER import TRANSFORMER
from Backend.UCLMR.runUCLMR import *
from Backend.SEN.runSEN import *
from Backend.TAN.runTAN import *
from Backend.Allada_Nandakumar.runAllada_Nandakumar import *

dataset_names_dict = {
    "semEval2016": 1,
    "semEval2017": 2,
    "FNC": 3,
    "MPCHI": 4,
    "EmergentLite": 5,
    "MPQA": 6,
    "IBMDebator": 7,
    "Procon": 8,
    "VAST": 9,
    "covid":10
}

algorithmes_names_dict = [
    "LIU",
    "SEN",
    "TAN",
    "TRANSFORMER",
    "UCLMR",
    "ALLADA_NANDAKUMAR"
]

def get_dataset_name():
    keys=dataset_names_dict.keys()
    return keys

def get_algorithmes_names():
    return algorithmes_names_dict


def split_data_topic_based(df_before_spliting, train_percent):
    train_dataset = pd.DataFrame(columns=df_before_spliting.columns)
    test_dataset = pd.DataFrame(columns=df_before_spliting.columns)
    for topic in df_before_spliting.Claim.unique():
        tmp_df=df_before_spliting.copy()
        tmp_df=tmp_df[tmp_df['Claim']==topic]
        tmp_train_dataset, tmp_test_dataset = model_selection.train_test_split(tmp_df, train_size=train_percent, shuffle=False)
        train_dataset=train_dataset.append(tmp_train_dataset)
        test_dataset=test_dataset.append(tmp_test_dataset)

    return train_dataset, test_dataset

def get_unique_labels(df):
    return df.Stance.unique()



# the function recieve models array (strings), dataset_name, and the division percent to train and test
def start_Specific_Model(models, dataset_name, train_percent,df_extenal,type_ds):
    # try:

        df_train_records=0
        df_test_records=0

        type=""
        db = DataBase()
        # receive DF
        dataset_names_dict = {
            "semEval2016" : 1,
            "semEval2017" : 2,
            "FNC" : 3,
            "MPCHI" : 4,
            "EmergentLite" : 5,
            "MPQA" : 6,
            "IBMDebator" : 7,
            "Procon" :8,
            "VAST" :9,
            "covid":10
        }

        if df_extenal is not None and dataset_name is None:
            df=df_extenal.copy()
            dataset_name=str(db.get_index())


        else:
            dataset_id = dataset_names_dict[dataset_name]

            df = db.get_dataset(dataset_id)
        index_models = ''
        for name in models:
            index_models = index_models + name + " "
        isExistInRequest = db.get_record_from_request(index_models, dataset_name, train_percent)
        if ( isExistInRequest is not None):
            return isExistInRequest
        # get unique labels
        labels = get_unique_labels(df)
        num_of_labels = len(labels)

        # split df to df_train and df_test
        train_percent = train_percent / 100

        if dataset_name == "FNC":
            df_train, df_test = model_selection.train_test_split(df, train_size=train_percent, shuffle=False)
            df_train_records = df_train.shape[0]
            df_test_records = df_test.shape[0]
            type='headline'
        elif dataset_name == "EmergentLite" or dataset_name =='IBMDebator' or dataset_name =='Procon' or  dataset_name =='VAST':
            df_train, df_test = model_selection.train_test_split(df, train_size=train_percent, random_state=42)
            type='headline'
            df_train_records = df_train.shape[0]
            df_test_records = df_test.shape[0]

        elif dataset_name == "semEval2016" or dataset_name == "semEval2017" or dataset_name == "MPCHI" or dataset_name == "MPQA"  or dataset_name=='covid':
            # df_train, df_test = model_selection.train_test_split(df, train_size=train_percent, random_state=42)
            df_train, df_test = split_data_topic_based(df, train_percent)
            df_train_records=df_train.shape[0]
            df_test_records=df_test.shape[0]
            type='topic'
        else:

            if df_extenal is not None:

                if(type_ds=='headline_based'):
                    df_train, df_test = model_selection.train_test_split(df, train_size=train_percent, random_state=42)
                    df_train_records = df_train.shape[0]
                    df_test_records = df_test.shape[0]
                else:
                    df_train, df_test = split_data_topic_based(df, train_percent)
                    df_train_records = df_train.shape[0]
                    df_test_records = df_test.shape[0]
            else:
                df_train, df_test = model_selection.train_test_split(df, train_size=train_percent, random_state=42)
                df_train_records = df_train.shape[0]
                df_test_records = df_test.shape[0]

        results = {}

        # for each  model name in models array run the model
        for m_name in models:


            ###check if the run is saved

            isExist = db.get_record_from_result(m_name, dataset_name, train_percent)
            if(isExist.shape[0]>0):
                continue

            if m_name == "SEN":
                sen = SEN()
                start = datetime.now()
                y_test, y_pred = sen.run_SEN(df_train, df_test, labels, num_of_labels)
                end = datetime.now()
                time = (end-start).total_seconds()/60
            elif m_name == "UCLMR":
                uclmr = UCLMR()
                start = datetime.now()
                y_test, y_pred, all_prob = uclmr.run_UCLMR(df_train, df_test, labels, num_of_labels)
                end = datetime.now()
                time = (end-start).total_seconds()/60
            elif m_name == "TAN":
                tan = TAN()
                start = datetime.now()
                y_test, y_pred ,all_prob= tan.run_TAN(df_train, df_test, labels, num_of_labels)
                all_prob=np.array(all_prob)
                end = datetime.now()
                time = (end-start).total_seconds()/60
            elif m_name == "TRANSFORMER":
                transformer = TRANSFORMER()
                start = datetime.now()
                y_test, y_pred, all_prob = transformer.run_TRANSFORMER(df_train, df_test, labels, num_of_labels, dataset_name,str(train_percent))
                end = datetime.now()
                time = (end-start).total_seconds()/60
            elif m_name == "ALLADA_NANDAKUMAR":
                allada_nandakumar = Allada_Nandakumar()
                start = datetime.now()
                y_test, y_pred, all_prob = allada_nandakumar.run_Allada_Nandakumar(df_train, df_test, labels)
                end = datetime.now()
                time = (end-start).total_seconds()/60
            elif m_name == "LIU":
                liu = LIU()
                start = datetime.now()
                y_test, y_pred = liu.run_LIU(df_train,df_test, labels, num_of_labels)
                end = datetime.now()
                time = (end-start).total_seconds()/60

            # with open('sample_' + m_name + '.csv', 'w', newline='') as csvfile:
            #     fieldnames = ['pred', 'test']
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #     writer.writeheader()
            #     i = 0
            #     for pred_model in y_pred:
            #         if i< len(y_test):
            #             writer.writerow({'pred': pred_model, 'test': y_test[i]})


            # each model returns y_test and y_predict
            # calculate accuracy, confusion matrix, classification report


            #actual vs predict
            array_labels=""
            for label in labels:
                array_labels=array_labels+label+" "


            actual=""
            predict=""

            for label in labels:
                num=y_test.count(label)
                actual=actual+str(num)+","

            for label in labels:
                predict=predict+str(y_pred.count(label))+","

# print(get_one_stance("I think she is a nice woman",'Hillary Clinton'))


            # calculate accuracy
            acc = accuracy_score(y_test, y_pred)
            acc = float("{:.3f}".format(acc))
            # get classification report
            cr = sklearn.metrics.classification_report(y_test, y_pred)

            # plot confusion matrix
            PROJECT_ROOT = os.path.abspath(__file__)
            BASE_DIR = os.path.dirname(PROJECT_ROOT)
            cm_path = BASE_DIR + '\\DB\\ConfusionMatrix\\' + m_name + '_ ' + dataset_name + '_ ' + str(train_percent) + '.png'
            cm = confusion_matrix(y_test, y_pred,labels=labels)
            # plot_confusion_matrix(cm_path, cm, target_names=labels, normalize=False)
            target =labels
            target_string=""
            for t in target:
                target_string=target_string+t+","
            cm_strings=""
            for array in cm:
                l1 = list(array)
                for i in l1:
                    cm_strings=cm_strings+str(i)+","
                cm_strings=cm_strings+"\n"

            # plot ROC Curve and find roc_auc accuracy
            roc_acc = multiclass_roc_auc_score(y_test, y_pred)
            roc_acc = float("{:.3f}".format(roc_acc))
            roc_path = BASE_DIR + '\\DB\\ROC\\' + m_name + '_ ' + dataset_name + '_ ' + str(train_percent) + '.png'
            dict_tpr_fpr=plot_multiclass_roc(labels,y_test, all_prob, roc_path, n_classes=num_of_labels, figsize=(16, 10))

            dict_tpr_fpr_string = json.dumps(dict_tpr_fpr)

            # save results in dictionary
            results[m_name] = {}
            results[m_name]['accuracy'] = acc
            results[m_name]['class_report'] = cr
            # results[m_name]['cm_path'] = cm_path
            results[m_name]['roc_acc'] = roc_acc
            # results[m_name]['roc_path'] = roc_path

            #insert to result table the details
            # for name in models:

            db.insert_records_to_result(m_name,dataset_name,train_percent,results[m_name]['accuracy'], results[m_name]['class_report'],

            results[m_name]['roc_acc'],actual,predict,array_labels,cm_strings,target_string,df_train_records,df_test_records,dict_tpr_fpr_string,type,time)

        index = db.insert_records_request(index_models,dataset_name,train_percent)
        return index
    # except:
    #     return False


# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(path, cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
#     """
#     given a sklearn confusion matrix (cm), make a nice plot
#
#     Arguments
#     ---------
#     cm:           confusion matrix from sklearn.metrics.confusion_matrix
#
#     target_names: given classification classes such as [0, 1, 2]
#                   the class names, for example: ['high', 'medium', 'low']
#
#     title:        the text to display at the top of the matrix
#
#     cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
#                   see http://matplotlib.org/examples/color/colormaps_reference.html
#                   plt.get_cmap('jet') or plt.cm.Blues
#
#     normalize:    If False, plot the raw numbers
#                   If True, plot the proportions
#
#     Usage
#     -----
#     plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
#                                                               # sklearn.metrics.confusion_matrix
#                           normalize    = True,                # show proportions
#                           target_names = y_labels_vals,       # list of names of the classes
#                           title        = best_estimator_name) # title of graph
#
#     Citiation
#     ---------
#     http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#
#     """
    import itertools

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def plot_multiclass_roc(labels,y_test, y_pred, path, n_classes, figsize=(17, 6)):
    # # structures
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    dict_fpr_tpr=[]

    if len(labels)>2:
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test_dummies = lb.transform(y_test)
        # y_pred_dummies = lb.transform(y_pred)

        y = label_binarize(y_test, classes=labels)
        n_classes = y.shape[1]
    else:

        encoder = LabelEncoder()
        encoder.fit(y_test)
        transfomed_label = encoder.transform(y_test)

        y_test_dummies = (transfomed_label[:, None] != np.arange(2)).astype(int)


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        arr=[]
        length_arr=int(len(fpr[i])/5)
        # for j in range(len(fpr[i])):
        j=0
        arr.append([round(fpr[i][0], 2),round(tpr[i][0], 2)])
        arr.append([round(fpr[i][j+length_arr], 2), round(tpr[i][j+length_arr], 2)])
        arr.append([round(fpr[i][j+2*length_arr], 2), round(tpr[i][j+2*length_arr], 2)])
        arr.append([round(fpr[i][j+3*length_arr], 2), round(tpr[i][j+3*length_arr], 2)])
        arr.append([round(fpr[i][len(fpr[i])-1], 2), round(tpr[i][len(fpr[i])-1], 2)])


        area = str(round(roc_auc[i], 2))
        name = labels[i].upper() + " <br> Area=" + area
        dict_fpr_tpr.append({'name': name, 'data': arr, 'area': round(roc_auc[i], 2)})

    # y_test_labels = np.unique(y_test)
    # if (n_classes == 2):
    #     y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    #     y_pred_dummies = pd.get_dummies(y_pred, drop_first=False).values
    #
    #
    #     if(len(y_pred_dummies[0]) == 1):
    #         y = y_test_dummies
    #
    #         for i in range(len(y)):
    #             y[i][0] = 1
    #             y[i][1] = 0
    #         y_pred_dummies = y
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred_dummies[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     arr=[]
    #     for j in range(len(fpr[i])):
    #         arr.append([round(fpr[i][j], 2),round(tpr[i][j], 2)])
    #     area = str(round(roc_auc[i], 2))
    #     name = y_test_labels[i].upper() + " <br> Area=" + area
    #     dict_fpr_tpr.append({'name': name, 'data': arr, 'area': round(roc_auc[i], 2)})

    # roc for each class
    # fig, ax = plt.subplots(figsize=figsize)
    # ax.plot([0, 1], [0, 1], 'k--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_title('ROC Curve')
    # for i in range(n_classes):
    #     ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %s' % (roc_auc[i], y_test_labels[i]))
    # ax.legend(loc="best")
    # ax.grid(alpha=.4)
    # #sns.despine()
    # #plt.show()
    # plt.savefig(path)

    return dict_fpr_tpr

#the function recieves a sentence and claim and returns its stance
def get_one_stance(sentence, claim,model_name,model_TRANSFORMER,train):
    dataset_id = dataset_names_dict["semEval2016"]
    db = DataBase()
    # df = db.get_dataset(dataset_id)

    if(db.get_record_from_Stance_Result(claim,sentence,model_name)).shape[0]>0:
        pred=db.get_record_from_Stance_Result(claim,sentence,model_name).iloc[0]['Stance']
        return pred
    else:
        # get unique labels
        # labels = get_unique_labels(df)
        labels=['AGAINST','FAVOR','NONE']
        num_of_labels = len(labels)
        if(model_name=='TRANSFORMER'):
            ts = TRANSFORMER()
            d = {'Claim': [claim], 'Sentence': [sentence], 'Stance': ['AGAINST']}
            df = pd.DataFrame(data=d)
            y_pred = ts.run_one_sen(None, df,labels,num_of_labels,"semEval2016",model_TRANSFORMER,train)
            db.insert_to_Stance_Result(claim,sentence,str(y_pred),model_name)
            return y_pred
        if(model_name=='TAN'):
            tan=TAN()
            y_pred=tan.get_one_stance(sentence,claim,None)
            db.insert_to_Stance_Result(claim,sentence,str(y_pred),model_name)

            return y_pred

# models = list()
# models.append("TRANSFORMER")
# start_Specific_Model(models, "Procon", 60, None, None)

# print(get_one_stance("I think she is a nice woman",'Hillary Clinton'))


