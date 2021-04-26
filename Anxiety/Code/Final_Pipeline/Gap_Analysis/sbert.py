import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')

#Loading all dataframes
#dfj=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Academic/topic_words_j2012.csv")
dfj=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Academic/topic_words_j2015.csv")
dfr12=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Reddit/topic_words_r2012.csv")
#dfr13=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Reddit/topic_words_r2013.csv")
#dfr14=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Reddit/topic_words_r2014.csv")
#dfr15=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Reddit/topic_words_r2015.csv")

# Corpus with example sentences
corpus = dfj['Most_freq_words'].tolist()
c_ids=dfj['Topic_ID'].tolist()
r_ids=dfr12['Topic_ID'].tolist()

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries=dfr12['Most_freq_words'].tolist()

#Creating dataframe for result
df_res=pd.DataFrame(columns=['corpus','corpus_year','query_source','query_year','query','query_topicID','best_match','best_match_score','best_match_topicID'])
df_res['query_topicID']=r_ids

#print(len(df_res_j12r12['corpus']))

# Find the closest topic of the corpus for each query sentence based on cosine similarity
top_k = min(1, len(corpus))
i=0
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("Topic ID of query:", queries.index(query))
    df_res['query_topicID'][i]=queries.index(query)
    df_res['query'][i]=query
    print("\nmost similar topic in corpus (Academic 2013):")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))
        df_res['best_match'][i]=corpus[idx]
        df_res['best_match_score'][i]=score.item()
        print("Topic ID of most similar topic in corpus:", c_ids[idx])
        df_res['best_match_topicID'][i]=c_ids[idx]
    i+=1

n=len(df_res['query_topicID'])
for i in range(n):
    df_res['corpus'][i]='Academic'
    df_res['corpus_year'][i]=2015
    df_res['query_source'][i]='Reddit'
    df_res['query_year'][i]=2012
df_res.to_csv("GA_j15r12.csv")
