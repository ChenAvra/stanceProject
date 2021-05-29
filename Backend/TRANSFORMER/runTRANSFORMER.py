
from .pred import Pred, Pred_one_sentence


class TRANSFORMER:
    #return pred_test and y_test
    def run_TRANSFORMER(self, df_train, df_test, labels, num_of_labels,dataset_name,train_percent):
        return Pred(df_train, df_test, labels, num_of_labels,dataset_name,train_percent)


    def run_one_sen(self, df_train, df_test, labels, num_of_labels,dataset_name,model_TRANSFORMER,train):
        return Pred_one_sentence(df_train, df_test, labels, num_of_labels,dataset_name,model_TRANSFORMER,train)
