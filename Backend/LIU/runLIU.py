from sklearn import model_selection

from .main import runLIU
import pandas as pd
class LIU:

    def run_LIU(self, df_train, df_test, labels, num_of_labels):
        return runLIU(df_train, df_test, labels, num_of_labels)

# def split_data_topic_based(df_before_spliting, train_percent):
#     train_dataset = pd.DataFrame(columns=df_before_spliting.columns)
#     test_dataset = pd.DataFrame(columns=df_before_spliting.columns)
#     for topic in df_before_spliting.Claim.unique():
#         tmp_df=df_before_spliting.copy()
#         tmp_df=tmp_df[tmp_df['Claim']==topic]
#         tmp_train_dataset, tmp_test_dataset = model_selection.train_test_split(tmp_df, train_size=train_percent, shuffle=False)
#         train_dataset=train_dataset.append(tmp_train_dataset)
#         test_dataset=test_dataset.append(tmp_test_dataset)
#
#     return train_dataset, test_dataset
# df = pd.read_csv("C:\\Users\\User\\Desktop\\Systance\\stanceProject\\Backend\\DB\\semEval2016.csv", header = None, names=['Claim','Sentence', 'Stance'])
# labels = list(set(df.Stance))
# labels_length = len(labels)
# df_train, df_test = split_data_topic_based(df, 0.8)
# LIU().run_LIU(df_train, df_test, labels, labels_length)