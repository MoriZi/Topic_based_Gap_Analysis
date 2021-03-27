#from config import *
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
#import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import string
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

'''
Make a new directory if the directory does not exist
'''
def make_dir(path):
    '''
    :param path: string, the path to save data.
    :return path: string, the path to save data.
    '''
    if os.path.isdir(path) is False:
        os.mkdir(path)


'''
Text mining for string
'''


def preprocess(text):
    '''
    :param text: string, the content of the document.
    :return result: a list of words that removed stopwords, tolower and lemmatized.
    '''

    result = []
    for t,v in nltk.pos_tag(gensim.utils.simple_preprocess(text)):
        if t not in gensim.parsing.preprocessing.STOPWORDS:
                result.append(WordNetLemmatizer().lemmatize(t.lower()))
    return result

'''
def preprocess(df):
    tokenizer=RegexpTokenizer(r'\w+')
    df['Title']=df['Title'].astype(str)
    df['Text']=df['Text'].astype(str)
    df['Title']=df['Title'].apply(lambda x: tokenizer.tokenize(x.lower()))
    df['Text']=df['Text'].apply(lambda x: tokenizer.tokenize(x.lower()))
    stop=stopwords.words('english')
    df['Text']=df['Text'].apply(lambda x: [item for item in x if item not in stop])
    df['Title']=df['Title'].apply(lambda x: [item for item in x if item not in stop])
    lemmatizer = WordNetLemmatizer()
    df['Text']=df['Text'].apply(lambda x: lemmatizer.lemmatize(str(x)))
    df['Title']=df['Title'].apply(lambda x: lemmatizer.lemmatize(str(x)))
    return df
'''

'''
Text mining for df
Create dictionary with df
'''

def process_dict(df):
    '''
    :param df: a pandas.core.frame.DataFrame.
    :return dictionary: a gensim.corpora.dictionary.Dictionary. Unique words in whole corpus.
    :return processed_docs: a pandas.core.series.Series, each row is a list of words of the doc after text mining.
    '''
    df['Text']=df['Text'].astype(str)
    processed_docs = df['Text'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    return dictionary, processed_docs

'''
Create bag of words
'''

def bow(dictionary, processed_docs):
    '''
    :param dictionary: a gensim.corpora.dictionary.Dictionary. Unique words in whole corpus.
    :param processed_docs: a pandas.core.series.Series, each row is a list of words of the doc after text mining
    :return bow_corpus: a list of lists. Each sub list is all unique word vectors and word counts of a doc.
                        Example: [[(0, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), ...

    '''
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    return bow_corpus


'''
Calculate low_tfidf_words, a list of word_id with LOW tfidf_score, so as to remove commonly used words later
'''

'''
1). create 2 dictionaries: total_word_count and DictDocFreq
'''
def tf_df(bow_corpus, dictionary):

    '''
    :param bow_corpus: a list of lists. Each sub list is all unique word vectors and word counts of a doc.
    :param dictionary: a gensim.corpora.dictionary.Dictionary. Unique words in whole corpus.
    :return total_word_count: a dictionary of {word_id: word_count, word_id: word_count, ... }.
                              Here word_count means the count of frequency of word appearance throughout all docs.
    :return DictDocFreq: a dictionary of {word_id: count, word_id: count, ... }.
                         Here count means the count of frequency of word appearance in each doc
                         (i.e. the word appeared in how many docs. Appear counts 1 and not appear counts 0).
    '''
    # create 2 empty dictionaries
    total_word_count = defaultdict(int)
    DictDocFreq = {}

    # calculate total_word_count
    for word_id, word_count in itertools.chain.from_iterable(bow_corpus):
        total_word_count[word_id] += word_count

    # calculate DictDocFreq
    for c in dictionary:
        count = 0
        for doc in bow_corpus:
            for word in doc:
                if word[0] == c:
                    count += 1
                    continue
            DictDocFreq[c] = count
    return total_word_count, DictDocFreq

'''
2). Create sorted_TFIDF: a dictionary of {word_id, tfidf_score} with tfidf_score in descending order.
tfidf_score is calculated based on an equation with total_word_count, DictDocFreq and len(bow_corpus) as parameters
'''

def sort_tfidf(bow_corpus, total_word_count, DictDocFreq):

    '''
    :param bow_corpus: a list of lists. Each sub list is all unique word vectors and word counts of a doc.
    :param total_word_count: a dictionary of {word_id: word_count, word_id: word_count, ... }.
                             Here word_count means the count of frequency of word appearance throughout all docs.
    :param DictDocFreq: a dictionary of {word_id: count, word_id: count, ... }.
                        Here count means the count of frequency of word appearance in each doc.
    :return sorted_TFIDF: a dictionary of {word_id, tfidf_score} with tfidf_score in descending order.
                          tfidf_score is calculated based on an equation with total_word_count, DictDocFreq and len(bow_corpus) as parameters.
    '''

    # create the empty dictionary and list
    dictTFIDF = {}

    D = len(bow_corpus)
    count = 0
    for a in total_word_count:
        count += 1

        # An intermediate dictTFIDF is calcualted based on an equation with total_word_count, DictDocFreq, and len(bow_corpus) as input.
        dictTFIDF[a] = int(total_word_count[a]) * np.log2(D / int(DictDocFreq[a]))

        # sort dictTFIDFbased on tfidf_score from high to low
        sorted_TFIDF = sorted(dictTFIDF.items(), key=lambda w: w[1], reverse=True)

    return sorted_TFIDF

'''
3). create 1 list: low_tfidf_words, it is a list of word_id with low tfidf score
with v calculated based on x - a parameters set by the user and len(dictionary) - the number of unique words
'''

def get_low_tfidf_words(x, dictionary, sorted_TFIDF):
    '''
    :param x: float, an argument set by user.
    :param dictionary: a gensim.corpora.dictionary.Dictionary. Unique words in whole corpus.
    :param sorted_TFIDF: a dictionary of {word_id, tfidf_score} with tfidf_score in descending order.
    :return low_tfidf_words: A list of word_id with LOW tfidf_score.
    '''
    low_tfidf_words = []

    v = int(x * len(dictionary))
    for a in sorted_TFIDF[-v:]:
        low_tfidf_words.append(a[0])

    return low_tfidf_words


'''
Filter out lest frequently used words
'''

def filter_least_frequent(dictionary, processed_docs, no_below_percent, keep_n):
    '''
    :param dictionary: a gensim.corpora.dictionary.Dictionary. Unique words in whole corpus.
    :param processed_docs: a pandas.core.series.Series, each row is a list of words of the doc after text mining.
    :param no_below_percent: float, an argument set by user.
    :param keep_n: int, an argument set by user.
    :return dict_least_freq_filtered: a gensim.corpora.dictionary.Dictionary, with lest frequently used words filtered out.
    '''

    # duplicate dictionary
    dict_least_freq_filtered = dictionary

    # filter out lest frequently used words based on count of frequency
    no_below = int(len(processed_docs) * no_below_percent)
    dict_least_freq_filtered.filter_extremes(no_below = no_below, keep_n = keep_n)

    print('There are {} unique words in the dictionary, {} remain after filtering out lest frequent.'\
          .format(len(dictionary), len(dict_least_freq_filtered)))

    return dict_least_freq_filtered

'''
Filter out most commonly used words
'''
def filter_most_common(dictionary, low_tfidf_words):

    '''
    :param dictionary: a gensim.corpora.dictionary.Dictionary.
    :param low_tfidf_words: A list of word_id with LOW tfidf_score.
    :return: a gensim.corpora.dictionary.Dictionary, with most commonly used words filtered out.
    '''
    # filter out most commonly word based on low_tfidf_words
    dictionary.filter_tokens(bad_ids = low_tfidf_words)

    print('{} remain after filtering out most commonly used words based on tfidf scores.'\
          .format(len(dictionary)))

    return dictionary


'''
Paste all words of a topic predicted by the lda_model to one dataframe
'''
def combine_topic_top_words(lda_model, num_topics, num_words):
    topics = {}
    for topic_id, word_prob_pairs in lda_model.show_topics(num_topics, num_words, log=False, formatted=False):
        topic_top_words = ''
        for word_prob_pair in word_prob_pairs:
            topic_top_words += word_prob_pair[0] + ' '
        topics[topic_id] = topic_top_words
    topic_combined_top_words = pd.DataFrame(topics.items(), columns=['topic_id', 'combined_top_words'])
    return topic_combined_top_words


'''
Calculate the count and mean of probability of each doc's predicted topic probability
This would indicate the topic trends among all docs in the directory, i.e. in the year
'''

def topic_doc_count_n_mean(lda_transformed_corpus, num_topics, bow_corpus_TFIDFfiltered):

    # create an empty dictionary with key 0 to num_topics-1
    DictTopicAccumulator = {}
    DictTopicCounter = {}
    for i in list(range(0, num_topics)):
        DictTopicAccumulator[i] = 0

    # initializing iterate of doc from 0
    doc = 0
    accumulated_prob_of_topic = 0
    #count_doc_in_topic = 0

    # iterate all doc_lda
    for doc_lda in lda_transformed_corpus:

        # print doc_id
        print("doc_id: " + str(doc))

        '''
        Explanation of `for tuple_topic_prob in doc_lda`:
        #### Function:
        Add and iterate the value to the empty dictionary DictTopicAccumulator. In each iteration,

              dictionary's key = the first element in tuple (topic, probability), i.e. the topic_id
              each dictionary key's value = the second element in the tuple (topic, probability), i.e. the topic_prob

        #### Output: DictTopicAccumulator

        A collection of:
            {(topic_id, sum_of_probability_among_all_documents),
             (topic_id, sum_of_probability_among_all_documents),
             (topic_id, sum_of_probability_among_all_documents),
             ...
        }
        '''

        # iterate of topic with probability of each doc
        for tuple_topic_prob in doc_lda:

            # print topic_id that the doc_id is predicted as
            print('Probability of this doc belongs to topic_id {} is {}: '. \
                  format(str(tuple_topic_prob[0]), str(tuple_topic_prob[1])))

            accumulated_prob_of_topic += tuple_topic_prob[1]
            # count_doc_in_topic += 1

            DictTopicAccumulator[tuple_topic_prob[0]] += tuple_topic_prob[1]  # , count_doc_in_topic)

            if tuple_topic_prob[0] not in DictTopicCounter.keys():
                DictTopicCounter[tuple_topic_prob[0]] = 1
            else:
                DictTopicCounter[tuple_topic_prob[0]] += 1

            print('accumulated_prob_of_topic: {}'. \
                  format(str(accumulated_prob_of_topic)))

            print('\n---------\n')

        doc += 1
        if doc >= len(bow_corpus_TFIDFfiltered):
            break

    topic_doc_count = pd.DataFrame(DictTopicCounter.items(), columns=['topic_id', 'doc_count'])
    topic_accumulated_probability = pd.DataFrame(DictTopicAccumulator.items(),
                                                 columns=['topic_id', 'accumulated_probability'])
    topic_prob = pd.merge(topic_accumulated_probability, topic_doc_count)
    topic_prob['probability_per_doc'] = topic_prob['accumulated_probability'] / topic_prob['doc_count']

    return topic_prob


'''
Merge topic_combined_top_words with topic_prob, save top_words result to '../result/topic/'
'''

def Topic_CombinedWord_ProbTopic(topic_combined_top_words, topic_prob, source, year, num_topics):

    result = pd.merge(topic_combined_top_words, topic_prob, # how = 'left',
                       on='topic_id')

    # create 2 new columns
    result['source'] = source
    result['year'] = year

    # save results
    result.to_csv(source + '_' + str(num_topics) + '_topics' + '.csv', index=False)
    return result

'''
Merge result as for different num_topics what are the relative coherence, perplexity
'''
def log(num_topics, coherence, perplexity):

    d = {
        'num_topics': [num_topics],
        'coherence': [coherence],
        'perplexity': [perplexity]
    }

    log_df = pd.DataFrame(data=d)
    return log_df

'''
Compute lda_model, u_mass coherence, perplexity, and top words for various number of topics
'''
def lda(dict_tfidf_filtered, bow_corpus_TFIDFfiltered,
        passes, workers, iterations,
        dictionary, dict_least_freq_filtered,
        num_words,
        data_directory,
        save_model_dir,
        start, limit, step,
        save_performance_dir):

    '''
    :param dict_tfidf_filtered: a gensim.corpora.dictionary.Dictionary, with both lest frequently and most commonly used words filtered out.
    :param bow_corpus_TFIDFfiltered: the second bag of words, created after lest frequently and most commonly used words  were filtered out.
    :param passes: an argument set by user. Used in Building LDA model.
    :param workers: an argument set by user. Used in Building LDA model.
    :param iterations: an argument set by user. Used in Building LDA model.
    :param dictionary: a gensim.corpora.dictionary.Dictionary. Unique words in whole corpus.
    :param dict_least_freq_filtered: a gensim.corpora.dictionary.Dictionary, with lest frequently used words filtered out.
    :param num_words: an argument set by user. How many top words in each topic to save as result of LDA topic modeling.
    :param data_directory: an argument set by user. The path where the data were read in.
    :param save_model_dir: an argument set by user. The path to save the LDA model.
    :param start: an argument set by user. Start of num_topics to test.
    :param limit: an argument set by user. Limit of num_topics to test.
    :param step: an argument set by user. Step of num_topics to test.
    :param save_performance_dir: an argument set by user. The path to save the performance result.
    :return: Nothing
    '''

    source = data_directory.split("/")[-3]
    year = data_directory.split("/")[-2]
    log_df_list = []

    print('Domain: ' + source)
    print('year: ' + year)

    for num_topics in range(start, limit, step):
        print('num_topics: ' + str(num_topics))

        lda_model = gensim.models.LdaMulticore(corpus=bow_corpus_TFIDFfiltered,
                                               num_topics=num_topics,
                                               id2word=dict_tfidf_filtered,
                                               passes=passes,
                                               workers=workers,
                                               iterations=iterations)

        #save_model_path = save_model_dir + source + '_' + year + '_' + str(num_topics) + '_topics/'
        save_model_path=save_model_dir

        make_dir(save_model_path)
        lda_model.save(save_model_path + 'lda')
        print('Saved LDA model to: ' + save_model_path)

        # Collect results of LDA model
        topic_combined_top_words = combine_topic_top_words(lda_model, num_topics, num_words)

        # Build tfidf model
        tfidf = gensim.models.TfidfModel(bow_corpus_TFIDFfiltered)

        # Get corpus_tfidf
        corpus_tfidf = tfidf[bow_corpus_TFIDFfiltered]

        # Calculate Coherence of the lda_model by building a CoherenceModel
        cm = gensim.models.coherencemodel.CoherenceModel(model=lda_model,
                                                         corpus = corpus_tfidf,
                                                         coherence='u_mass')
        # Obtain the coherence of the lda_model
        coherence = cm.get_coherence()
        print('coherence: ' + str(coherence))

        # Calculate Perplexity of the lda_model

        perplexity = lda_model.log_perplexity(bow_corpus_TFIDFfiltered)
        print('perplexity: ' + str(perplexity))


        '''
        Note: lda_transformed_corpus is topic probability distribution for bow_corpus_TFIDFfiltered, len(bow_corpus_TFIDFfiltered)
        Type: gensim.interfaces.TransformedCorpus
        it contains len(bow_corpus_TFIDFfiltered) lists, each list contains one or more tuples [(topic, probability),(topic, probability), ...]
        '''

        lda_transformed_corpus = lda_model[bow_corpus_TFIDFfiltered]

        # Calculate the count and mean of probability of each doc's predicted topic probability

        print("each doc's predicted topic probability:\n")

        topic_prob = topic_doc_count_n_mean(lda_transformed_corpus, num_topics, bow_corpus_TFIDFfiltered)

        # Merge topic_combined_top_words with topic_prob
        result = Topic_CombinedWord_ProbTopic(topic_combined_top_words, topic_prob, source, year, num_topics)

        # Write coherence, perplexity for each num_topics of a LDA model
        log_df = log(num_topics, coherence, perplexity)

        log_df_list.append(log_df)

        print('===========\n')

    num_topics_performance = pd.concat(log_df_list)

    num_topics_performance['source'] = source
    num_topics_performance['year'] = year
    num_topics_performance['dict_size_orig'] = len(dictionary)
    num_topics_performance['dict_size_filtering_out_lest_freq_word'] = len(dict_least_freq_filtered)
    num_topics_performance['dict_size_filtering_out_most_freq_word'] = len(dict_tfidf_filtered)
    num_topics_performance.to_csv('Result'+ str(year)+'.csv', index=False)
    #num_topics_performance.to_csv(save_performance_dir + source + '_' + year + '.csv', index=False)

    return
