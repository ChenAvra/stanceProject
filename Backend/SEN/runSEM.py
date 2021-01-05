from .SEM import train_model_topic_based
from .SEM import train_model_headline_based

class SEN:

    def run_SEN(self, df_train, df_test, labels, num_of_labels):
        return train_model_topic_based(df_train, df_test, labels, num_of_labels)