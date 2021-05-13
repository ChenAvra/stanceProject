from .model import Pred

class Allada_Nandakumar:

    def run_Allada_Nandakumar(self, df_train, df_test, labels):
        return Pred(df_train, df_test, labels)