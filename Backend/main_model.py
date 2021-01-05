import os

import sklearn
import sklearn.model_selection as model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

from Backend.DB.DBManager import *
from Backend.UCLMR.runUCLMR import *
from Backend.SEN.runSEM import *
from Backend.TAN.runTAN import *


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

# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(path, cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
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
    # plt.show()
    # plt.close()

def get_unique_labels(df):
    return df.Stance.unique()


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def plot_multiclass_roc(y_test, y_pred, path, n_classes, figsize=(17, 6)):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_dummies = lb.transform(y_test)
    y_pred_dummies = lb.transform(y_pred)

    y_test_labels = np.unique(y_test)

    # y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    # y_pred_dummies = pd.get_dummies(y_pred, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred_dummies[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %s' % (roc_auc[i], y_test_labels[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    #sns.despine()
    #plt.show()
    plt.savefig(path)


# the function recieve models array (strings), dataset_name, and the division percent to train and test
def start_Specific_Model(models, dataset_name, train_percent):

    # receive DF
    dataset_names_dict = {
        "semEval2016" : 1,
        "semEval2017" : 2,
        "FNC" : 3,
        "MPCHI" : 4,
        "EmergentLite" : 5,
    }

    dataset_id = dataset_names_dict[dataset_name]
    db = DataBase()
    df = db.get_dataset(dataset_id)

    # get unique labels
    labels = get_unique_labels(df)
    num_of_labels = len(labels)

    # split df to df_train and df_test
    train_percent = train_percent / 100

    if dataset_name == "FNC":
        df_train, df_test = model_selection.train_test_split(df, train_size=train_percent, shuffle=False)
    elif dataset_name == "semEval2016" or dataset_name == "semEval2017" or dataset_name == "MPCHI":
        # df_train, df_test = model_selection.train_test_split(df, train_size=train_percent, random_state=42)
        df_train, df_test = split_data_topic_based(df, train_percent)
    else:
        df_train, df_test = model_selection.train_test_split(df, train_size=train_percent, random_state=42)


    results = {}

    # for each  model name in models array run the model
    for m_name in models:
        if m_name == "SEN":
            sen = SEN()
            y_test, y_pred = sen.run_SEN(df_train, df_test, labels, num_of_labels)
        elif m_name == "UCLMR":
            uclmr = UCLMR()
            y_test, y_pred = uclmr.run_UCLMR(df_train, df_test, labels, num_of_labels)
        elif m_name == "TAN":
            tan = TAN()
            y_test, y_pred = tan.run_TAN(df_train, df_test, labels, num_of_labels)

        # each model returns y_test and y_predict
        # calculate accuracy, confusion matrix, classification report

        # calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        # get classification report
        cr = sklearn.metrics.classification_report(y_test, y_pred)

        # plot confusion matrix
        PROJECT_ROOT = os.path.abspath(__file__)
        BASE_DIR = os.path.dirname(PROJECT_ROOT)
        cm_path = BASE_DIR + '\\DB\\ConfusionMatrix\\' + m_name + '_ ' + dataset_name + '_ ' + str(train_percent) + '.png'
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm_path, cm, target_names=np.unique(y_test), title="Confusion Matrix", normalize=False)

        # plot ROC Curve and find roc_auc accuracy
        roc_acc = multiclass_roc_auc_score(y_test, y_pred)
        roc_path = BASE_DIR + '\\DB\\ROC\\' + m_name + '_ ' + dataset_name + '_ ' + str(train_percent) + '.png'
        plot_multiclass_roc(y_test, y_pred, roc_path, n_classes=num_of_labels, figsize=(16, 10))


        # save results in dictionary
        results[m_name] = {}
        results[m_name]['accuracy'] = acc
        results[m_name]['class_report'] = cr
        results[m_name]['cm_path'] = cm_path
        results[m_name]['roc_acc'] = roc_acc
        results[m_name]['roc_path'] = roc_path


    return results

#the function recieves a sentence and claim and returns its stance
def get_one_stance(sentence, claim, stance):
    tan = TAN()
    stance = tan.get_one_stance(sentence, claim, stance)
    return stance

# models = list()
# models.append("UCLMR")
# start_Specific_Model(models, "FNC", 66)

# print(get_one_stance("I think it's bad",'Are E-Cigarettes safe?','AGAINST'))