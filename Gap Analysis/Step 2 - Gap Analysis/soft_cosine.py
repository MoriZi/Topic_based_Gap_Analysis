from config import *
from topic import Topic
from gap_analysis_function import *
# ============================
#        I. Arguments
#        II. Main
# ============================


# ============================
#         I. Arguments
# ============================

parser = argparse.ArgumentParser()

# directory to read source 1
parser.add_argument("-data1", "--data_source_one_directory", default = '../result/Academic/', type = str,
                     help = "Read in results produced by topic modeling. All files in the dir would be read in and concat.")

# directory to read source 2
parser.add_argument("-data2", "--data_source_two_directory", default = '../result/NYT/', type = str,
                     help = "Read in results produced by topic modeling. All files in the dir would be read in and concat.")


# directory to save results
parser.add_argument("-savedir", "--save_directory", default = '../result/gap_analysis/', type = str,
                     help = "Directory where the results in gap analysis should be saved to.")

# to save the heatmap or not
parser.add_argument("-saveheatmap", "--saveheatmap", default = False, type = bool,
                   help = "Whether to a save heatmap plot among all topics.")



args = parser.parse_args()


# ============================
#        II. Main
# ============================

# 1. read in data
source1_df = Topic(data_directory = args.data_source_one_directory).df
source2_df = Topic(data_directory = args.data_source_two_directory).df


# 2. Make a dictionary and a corpus using top words of all topics from 2 domains
source1_word_list = source1_df['combined_top_words'].tolist()
source2_word_list = source2_df['combined_top_words'].tolist()
joined_list = source1_word_list + source2_word_list


docs = []
for combined_top_words in joined_list:
    docs.append(combined_top_words.split())
dictionary = gensim.corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]


# 3. Calculate Similarity Matrix using a wiki vocabulary and the dictionary.
# It will be used to to calculate soft cosine between 2 topics in function softcosine_among_topics()


# !Please download the model ONCE only
# w2v_model = api.load("fasttext-wiki-news-subwords-300")
# pickle.dump(w2v_model, open('../downloaded_model/w2v_model.sav', 'wb'))

w2v_model = pickle.load(open('../downloaded_model/w2v_model.sav', 'rb'))
similarity_matrix = w2v_model.similarity_matrix(dictionary)

# 4. Create a dict, which collects all soft cosine measure among all topics from the 2 sources
dict = softcosine_among_topics(source1_df, source2_df, dictionary, similarity_matrix)

# 5. convert dict to matrix
matrix = dict_to_matrix(dict, source2_df['label'],  figsize_x = 70, figsize_y = 65,
                        saveheatmap = args.saveheatmap, save_graph_name = 'heatmap')

print('matrix shape: ' + str(matrix.shape))

matrix = index_as_column(matrix, save_dir = args.save_directory, save_matrix_name = 'matrix')

# 6. convert the matrix into a long table
long = to_longtable(matrix, save_dir = args.save_directory, save_longtable_name = 'long')

# 7. plot the histgram of the distribution of soft cosine
plot_hist(long)

