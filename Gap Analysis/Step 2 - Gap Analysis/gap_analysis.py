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

# directory to read long table
parser.add_argument("-long", "--long", default = '../result/gap_analysis/long.csv', type = str,
                     help = "Read in long table.")

# directory to read source 1
parser.add_argument("-data1", "--data_source_one_directory", default = '../result/Academic/', type = str,
                     help = "Read in results produced by topic modeling. All files in the dir would be read in and concat.")

# directory to read source 2
parser.add_argument("-data2", "--data_source_two_directory", default = '../result/NYT/', type = str,
                     help = "Read in results produced by topic modeling. All files in the dir would be read in and concat.")


# threshold of soft cosine measure to decide which topics are the the same topic
parser.add_argument("-threshold", "--threshold", default = .3, type = float,
                   help = "Two topics with soft cosine measure >= threshold would be considered as same topic.")


# directory to save results
parser.add_argument("-savedir", "--save_directory", default = '../result/gap_analysis/', type = str,
                     help = "Directory where the results in gap analysis should be saved to.")




args = parser.parse_args()


# ============================
#        II. Main
# ============================

# 1. Read in the long table and topic data

long = pd.read_csv(args.long)

source1_df = Topic(data_directory = args.data_source_one_directory).df
source2_df = Topic(data_directory = args.data_source_two_directory).df

# 2. Use the args.threshold to find similar topics from source1 vs. source2
significant = long[long['softcosine'] > args.threshold]


# 3. Calculate gap year between each of 2 similar topics
significant_w_gap = calulate_gap_year(significant)

# 4. Merge the top words of each topic to the significant_w_gap

threshold_similar_topic_w_feature_gap = merge_gap_with_topwords(significant_w_gap, source1_df,
                                                      source2_df,
                                                      save_dir = args.save_directory,
                                                      save_gap_name = 'threshold_'+ str(args.threshold)+'_similar_topic_w_feature_gap')

print('Shape of threshold_' + str(args.threshold) + '_similar_topic_w_feature_gap:')
print(str(threshold_similar_topic_w_feature_gap.shape))



