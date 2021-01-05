from .SVM import train_model_topic_based
from .SVM import train_model_headline_based

class SVM:

    def run_SVM(self, df_train, df_test, labels, num_of_labels):
        return train_model_topic_based(df_train,df_test,labels,num_of_labels)