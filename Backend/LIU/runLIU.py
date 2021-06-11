from sklearn import model_selection

from .main import runLIU
import pandas as pd
class LIU:

    def run_LIU(self, df_train, df_test, labels, num_of_labels):
        return runLIU(df_train, df_test, labels, num_of_labels)
