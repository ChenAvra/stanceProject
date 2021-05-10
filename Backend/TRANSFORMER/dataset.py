import os
import pandas as pd

## Methods for loading both csvs into dataframes and merge them into a single Dataframe.

def readDataset(path):
    stances = pd.read_csv(os.path.join(path,'train_stances.csv'))
    # stances = pd.read_csv(os.path.join(path,'train_s_chen.csv'))

    bodies = pd.read_csv(os.path.join(path,'train_bodies.csv'))
    # bodies = pd.read_csv(os.path.join(path,'train_b_chen.csv'))

    return stances.merge(bodies,on='Body ID')

def readTestDataset(path):
    stances = pd.read_csv(os.path.join(path,'competition_test_stances.csv'))
    # stances = pd.read_csv(os.path.join(path,'test_s_chen.csv'))

    bodies = pd.read_csv(os.path.join(path,'competition_test_bodies.csv'))
    # bodies = pd.read_csv(os.path.join(path,'test_b_chen.csv'))

    return stances.merge(bodies,on='Body ID')