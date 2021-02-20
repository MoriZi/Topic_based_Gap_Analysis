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
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora
from gensim.matutils import softcossim

def softcosine_among_topics(df1, df2, dictionary, similarity_matrix):
    dict = {}
    for i1 in range(len(df1)):  # df1's each year each topic, split top words and do boc2bow with dictionary

        bow1 = dictionary.doc2bow(df1['combined_top_words'].iloc[i1].split())

        lst = []
        for i2 in range(len(df2)):  # df2 each year

            bow2 = dictionary.doc2bow(df2['combined_top_words'].iloc[i2].split())

            softcosine = softcossim(bow1, bow2, similarity_matrix)

            lst.append(softcosine)


        dict[df1['label'].iloc[i1]] = lst

    return dict


def dict_to_matrix(dict, columnname,
                   figsize_x = 70, figsize_y = 65, save_graph_name = 'heatmap',
                   saveheatmap = False):
    # to df
    matrix = pd.DataFrame.from_dict(dict, orient = 'index', columns = columnname)


    if saveheatmap == True:
        fig1, ax1 = plt.subplots(1, 1, figsize = (figsize_x, figsize_y))
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        sns.heatmap(matrix).figure.savefig( save_graph_name + ".png")
        plt.close(fig1)
    return matrix

def index_as_column(matrix, save_dir = "/home/smriti/Desktop/NLP/MITACS/GapAnalysis/", save_matrix_name = 'matrix'):

    # reset index
    matrix = matrix.reset_index()

    # remove index name
    matrix = matrix.rename_axis(None, axis=1)

    # rename first column
    matrix.rename(columns={matrix.columns[0]: 'source1_label'}, inplace=True)

    # save
    matrix.to_csv(save_dir + save_matrix_name + ".csv", index=False)

    return matrix


def to_longtable(matrix, save_dir = "/home/smriti/Desktop/NLP/MITACS/GapAnalysis/", save_longtable_name = 'long'):

    long = pd.melt(matrix, id_vars=['source1_label'],
                   var_name='source2_label', value_name='softcosine')

    long.to_csv(save_dir + save_longtable_name + ".csv", index=False)

    return long

def plot_hist(long, save_dir = "/home/smriti/Desktop/NLP/MITACS/GapAnalysis/", save_hist_name = 'hist_softcosine'):
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))
    ax2.hist(long['softcosine'], bins=50)
    plt.xlabel('Soft Cosine')
    plt.ylabel('Frequency')
    fig2.savefig(save_dir + save_hist_name + ".png")
    plt.close(fig2)
    return

def calulate_gap_year(significant):
    # expand columns from souce1_label
    #print("Here\n")
    significant = significant.join(significant['source1_label'].str.split('_', 4, expand=True))
    significant = significant.rename(columns={0: "source1_source", 1: "source1_year",
                                              2: "source1_topic_id", 3: "source1_topword"})
    # expand columns from souce2_label
    #print("Here too\n")
    significant = significant.join(significant['source2_label'].str.split('_', 3, expand=True))
    significant = significant.rename(columns={0: "source2_source", 1: "source2_year",
                                              2: "source2_topic_id", 3: "source2_topword"})

    # year columns to numeric
    print((significant))
    significant[["source1_year", "source2_year"]] = significant[["source1_year", "source2_year"]].apply(pd.to_numeric)

    # calculate gap
    significant['year_diff_source2_minus_source1'] = significant['source2_year'] - significant['source1_year']

    return significant


def merge_gap_with_topwords(significant, source1_df, source2_df, save_dir = "../result/gap_analysis/", save_gap_name='similar_topic_w_feature_gap'):

    similar_topic_w_feature_gap = pd.merge(significant, source1_df[['label',
                                                                    'combined_top_words',
                                                                    'probability_per_doc',
                                                                    'accumulated_probability',
                                                                    'doc_count'
                                                                    ]], left_on='source1_label', right_on='label')

    similar_topic_w_feature_gap = pd.merge(similar_topic_w_feature_gap,
                                           source2_df[['label',
                                                       'combined_top_words',
                                                       'probability_per_doc',
                                                       'accumulated_probability',
                                                       'doc_count'
                                                       ]],
                                           left_on='source2_label', right_on='label',
                                           suffixes=('_source1', '_source2'))

    similar_topic_w_feature_gap = similar_topic_w_feature_gap.drop(columns=['label_source1', 'label_source2'])

    similar_topic_w_feature_gap.to_csv(save_dir + save_gap_name + ".csv", index=False)

    return similar_topic_w_feature_gap
