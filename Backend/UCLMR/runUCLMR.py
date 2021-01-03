from .pred import Pred

class UCLMR:

    def run_UCLMR(self, df_train, df_test, labels, num_of_labels):
        return Pred(df_train, df_test, labels, num_of_labels)