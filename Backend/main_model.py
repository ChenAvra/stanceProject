import pandas as pd

def get_unique_labels(df):
    return  df.Stance.unique()

#the function recieve models array-strings, dataset_id,division percent to train and test
def start_Specific_Model(models,dataset_name,train_percent):
    accuracy=0
    #recieve DF
    #get unique labels
    #split df to df_train and df_test

    #for each  model name in models array run the model
    #each model returns y_test and y_predict

    #calculate accuracy -confusion matrix



    return accuracy