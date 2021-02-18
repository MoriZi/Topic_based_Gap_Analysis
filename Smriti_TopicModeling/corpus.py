from config import *
import glob
import pandas as pd
import nltk
import gensim
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import itertools
import os

'''
Create a Class Corpus, it's process_dict method can output dictionary, processed_docs
'''


class Corpus:

    # read in files and prepare data table,
    # produce self.data_text['content'], an one column df as corpus what would be used further
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

        # remove rows  with missing value
        self.df = self.df.dropna()

        # print data size

        print('Read in {} files, concatenated to a data table of {} rows/documents.' \
              .format(len(file_list), self.df.shape[0]))

        # rename column name
        self.df.columns = ['title', 'content']

        # only use the content - named 'data_text' column for topic modeling
        self.data_text = self.df.loc[:, ['content']]

        # customized requirement
        self.data_text['content'] = self.data_text['content'].str.replace('Bit coin', 'bitcoin')
        self.data_text['content'] = self.data_text['content'].str.replace('bit coin', 'bitcoin')
