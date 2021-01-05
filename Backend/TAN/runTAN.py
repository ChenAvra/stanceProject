
from .pred import Pred, get_predict_per_stance


class TAN:
    #return pred_test and y_test
    def run_TAN(self, df_train, df_test, labels, num_of_labels):
        return Pred(df_train, df_test, labels, num_of_labels)

    def get_one_stance(self, sentence, claim, stance):
        return get_predict_per_stance(sentence, claim, stance)
