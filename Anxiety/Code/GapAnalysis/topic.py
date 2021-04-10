from config import *
import argparse
import glob
import pandas as pd
import nltk
import gensim
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import itertools
import numpy as np
import os

'''
Create a Class Topic. It would be used in Step 2. gap_analyis
'''

class Topic:

    # Read in files and prepare data table
    def __init__(self, data_directory):

        file_list = list(glob.glob(data_directory + "*.csv*"))
        file_list.sort()

        # print out file list
        print('In directory {} '.format(data_directory))
        print('Found files: ')
        print(file_list)

        # if number_of_file = 0, raise warning
        if len(file_list) == 0:
            print('There is no file in input directory: ' + data_directory)
            raise ValueError

        # if number_of_file = 1, read the first and only element in the list
        if len(file_list) == 1:

            self.df = pd.read_csv(file_list[0], error_bad_lines=False)


        # if number_of_file >1, read and append all files
        else:
            self.df_list = []
            for f in file_list:
                df = pd.read_csv(f, error_bad_lines=False)
                self.df_list.append(df)
            self.df = pd.concat(self.df_list).reset_index(drop=True)


        # split combined_top_words column into two parts, join back to df
        self.df = self.df.join(self.df['Keywords'].str.split(' ', 1, expand=True))

        #keep only the first part (most_freq_word)
        #self.df = self.df.drop(columns=[1])
        self.df = self.df.rename(columns={0: "most_freq_word"})

        self.df['label'] = self.df['source'] + '_' + self.df['year'].map(str) + \
                           '_Topic' + self.df['Dominant_Topic'].map(str) + '_' + self.df['most_freq_word']

        # remove rows  with missing value
        self.df = self.df.dropna()

        # print out data size
        print('Read in {} files, concatenated to a data table of {} rows.' \
              .format(len(file_list), self.df.shape[0]))
