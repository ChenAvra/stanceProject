from .pred import Pred

class MLP:

    def run_MLP(self, df_train, df_test, labels, num_of_labels):
        return Pred(df_train, df_test, labels, num_of_labels)