from config import *
from corpus import Corpus
from topic_model_function import *
from collections import defaultdict
import itertools
import os

'''

Usage Example:

# for 1 year's data in one domain, try num_topics = [5, 10, 15]
python3 topic_model.py -data '../data/Academic/2008/' -start 5 -limit 20 -step 5 -save_performance '../result/topic_performance2/'

# for 10 year's data in one domain, try num_topics = [20, 30, 40, 50]
for i in '2008' '2009' '2010' '2011' '2012' '2013' '2014' '2015' '2016' '2017'; do python3 topic_model.py -data '../data/Academic/'$i'/' -start 20 -limit 60 -step 10 -save_performance '../result/topic_performance2/'; done

'''


# ============================
#        I. Arguments
#        II. Main
# ============================


# ============================
#         I. Arguments
# ============================

parser = argparse.ArgumentParser()

# directory to read and save
parser.add_argument("-data", "--data_directory", default = '../data/NYT/2017/', type = str,
                     help = "Directory to read in all input csv files")

parser.add_argument("-result", "--result_directory", default = '../result/', type = str,
                     help = "Directory to save results")

# how to filter out less frequently used words in .filter_extremes() method
parser.add_argument("-nobelow",
                    "--no_below_percent",
                    default = .01, type = float,
                    help = "How to filter out less frequently used words in .filter_extremes() method")

parser.add_argument("-keep_n", "--keep_n", default = 100000, type = int,
                   help = "How to filter out less frequently used words in .filter_extremes() method")


# how to filter out most commonly used words (i.e. words with low TF-IDF score)
parser.add_argument("-x", "--x", default = .2, type = float,
                   help = "How to filter out most commonly used words (i.e. words with low TF-IDF score). \
                           Keep only words with tfidf ranking <= x * len(dictionary)")

parser.add_argument("-passes", "--passes", default = 150, type = int,
                   help = "Parameters building LDA")

parser.add_argument("-workers", "--workers", default = 3, type = int,
                   help = "Parameters building LDA")

parser.add_argument("-iterations", "--iterations", default = 3000, type = int,
                   help = "Parameters building LDA")

parser.add_argument("-save_model_dir", "--save_model_dir", default = '/home/smriti/Desktop/NLP', type = str,
                   help = "Save model to the directory")

# how many top words in each topic to save
parser.add_argument("-words", "--num_words", default = 10, type = int,
                   help = "How many top words in each topic to save as result in topic modeling. \
                           Those words would be used as inputs in topic forecasting.")


# how many top words in each topic to save
parser.add_argument("-start", "--start", default = 20, type = int,
                   help = "Starting num_topics to test.")

# how many top words in each topic to save
parser.add_argument("-limit", "--limit", default = 30, type = int,
                   help = "Limit of num_topics to test.")

# how many top words in each topic to save
parser.add_argument("-step", "--step", default = 10, type = int,
                   help = "Step of num_topics to test.")



parser.add_argument("-save_performance", "--save_performance_dir", default = '/home/smriti/Desktop/NLP', type = str,
                   help = "Save performance of all num_topics to the directory")

args = parser.parse_args()


# ============================
#        II. Main
# ============================

# 1. read in data and make it a df
print('\n====Read in Data====\n')
df = Corpus(data_directory = args.data_directory).df

# 2. Text mining
print('\n====Text Mining====\n')
dictionary, processed_docs = process_dict(df)

# 3. Create the first bag of words - bow_corpus
bow_corpus = bow(dictionary, processed_docs)

# 4. Calculate low_tfidf_words
total_word_count, DictDocFreq = tf_df(bow_corpus, dictionary)

sorted_TFIDF = sort_tfidf(bow_corpus, total_word_count, DictDocFreq)

low_tfidf_words = get_low_tfidf_words(args.x, dictionary, sorted_TFIDF)

# 5. Filter out lest frequently used words
dict_least_freq_filtered = filter_least_frequent(dictionary, processed_docs, args.no_below_percent, args.keep_n)

# 6. Filter out most commonly used words
dict_tfidf_filtered = filter_most_common(dict_least_freq_filtered, low_tfidf_words)

# 7. Create the second bag of words - bow_corpus_TFIDFfiltered, created after lest frequently and most commonly used words were filtered out.
bow_corpus_TFIDFfiltered = bow(dict_tfidf_filtered, processed_docs)

# 8. LDA
print('\n====Find Best num_topics and LDA====\n')
lda(dict_tfidf_filtered, bow_corpus_TFIDFfiltered,
    args.passes,  args.workers, args.iterations,
    dictionary, dict_least_freq_filtered,
    num_words = args.num_words,
    data_directory = args.data_directory,
    save_model_dir = args.save_model_dir,
    start = args.start,
    limit = args.limit,
    step = args.step,
    save_performance_dir = args.save_performance_dir)
