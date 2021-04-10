import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')

#Loading all dataframes
dfj=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Academic/topic_words_j2015.csv")
#dfr12=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Reddit/topic_words_r2012.csv")
#dfr13=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Reddit/topic_words_r2013.csv")
#dfr14=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Reddit/topic_words_r2014.csv")
dfr15=pd.read_csv("/home/smriti/Desktop/NLP/MITACS/Anxiety/Data/CSV/Reddit/topic_words_r2015.csv")

# Corpus with example sentences
corpus = dfj['Most_freq_words'].tolist()
c_ids=dfj['Topic_ID'].tolist()

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries=dfr15['Most_freq_words'].tolist()
#q2=dfr13['Most_freq_words'].tolist()
#q3=dfr14['Most_freq_words'].tolist()
#q4=dfr15['Most_freq_words'].tolist()

# Find the closest topic of the corpus for each query sentence based on cosine similarity
top_k = min(1, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("Querying Reddit 2015 topics:\n")
    print("\n\n======================\n\n")
    print("Query:", query)
    print("Topic ID of query:", queries.index(query))
    print("\nmost similar topic in corpus (Academic 2015):")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))
        print("Topic ID of most similar topic in corpus:", c_ids[idx])
