import pandas as pd
import csv
import re
import os
import numpy as np
# from utils import utils
import random
from sklearn.model_selection import train_test_split
from .Preprocess import Preprocess
import matplotlib.pyplot as plt

# LABELS = []

class DataSet():

    def __init__(self, preprocess=False, labelsFromDB = ['agree', 'disagree', 'discuss', 'unrelated'], train_df=None, test_df=None):
        '''
        Reading in article data from ../Data folder and generate dataframes contain news data;

        :param preprocess: (Boolean) True, to clean data set. Deault=False
        '''
        self.LABELS=labelsFromDB
        self.__preprocess = preprocess  # if yes, clean data set
        self.__all = []
        self.__train_all, self.__val_all, self.__test_all = self.__reader(train_df, test_df)


    def __reader(self, train_df, test_df):
        '''

        :return:
        '''

        "read in training sets"

        # clean two data sets
        if(self.__preprocess):
            print('Cleaning training set...\n')
            train_df = Preprocess(train_df).preprocess(['Claim', 'Sentence'])
            print('\nCleaning testing set...\n')
            test_df = Preprocess(test_df).preprocess(['Claim', 'Sentence'])
        # re-order column index, and drop some columns
        train_df = train_df[['Claim','Sentence','Stance']]
        test_df = test_df[['Claim', 'Sentence', 'Stance']]
        # use target labels to uniformly split data set
        train_all, val_all = train_test_split(train_df, train_size=0.9, random_state=0, stratify=train_df['Stance'])

        train_all = np.array(train_all)
        val_all = np.array(val_all)
        test_all = np.array(test_df)

        return train_all, val_all, test_all

    def get_train(self):
        return self.__train_all

    def get_validation(self):
        return self.__val_all

    def get_test(self):
        return self.__test_all

    def plot_distribution(self):
        # Generate data on commute times.
        df = self.__all['stance'].value_counts()


        # ax = df.plot().hist(grid=True, bins=4, rwidth=0.9,
        #                    color='#607c8e')
        df.plot(kind='bar', x=[1,2,3,4])

        plt.title('label distribution')
        plt.xlabel('class')
        plt.ylabel('counts')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
if __name__ == "__main__":
    DataSet()

