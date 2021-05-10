import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')

rls=['topic_words_r2012.csv', 'topic_words_r2013.csv', 'topic_words_r2014.csv', 'topic_words_r2015.csv', 'topic_words_r2016.csv', 'topic_words_r2017.csv', 'topic_words_r2018.csv', 'topic_words_r2019.csv', 'topic_words_r2020.csv']
mls=['topic_words_m2012.csv', 'topic_words_m2013.csv', 'topic_words_m2014.csv', 'topic_words_m2015.csv', 'topic_words_m2016.csv', 'topic_words_m2017.csv', 'topic_words_m2018.csv', 'topic_words_m2019.csv', 'topic_words_m2020.csv']
years=[2012,2013,2014,2015,2016,2017,2018,2019,2020]

for i in rls:
    for j in mls:
        #Loading all dataframes
        w=mls.index(j)
        print(j)
        fname1='/home/smriti/Smriti/MITACS/Anxiety/Data/CSV/Reddit/'+i
        fname2='/home/smriti/Smriti/MITACS/Anxiety/Data/CSV/Medium/'+j
        dfj=pd.read_csv(fname1)
        dfr12=pd.read_csv(fname2)

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
        m=0
        for query in queries:
            query_embedding = embedder.encode(query, convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            #print("\n\n======================\n\n")
            #print("Query:", query)
            #print("Topic ID of query:", queries.index(query))
            df_res['query_topicID'][m]=queries.index(query)
            df_res['query'][m]=query
            #print("\nmost similar topic in corpus:")

            for score, idx in zip(top_results[0], top_results[1]):
                #print(corpus[idx], "(Score: {:.4f})".format(score))
                df_res['best_match'][m]=corpus[idx]
                df_res['best_match_score'][m]=score.item()
                #print("Topic ID of most similar topic in corpus:", c_ids[idx])
                df_res['best_match_topicID'][m]=c_ids[idx]
            m+=1

        n=len(df_res['query_topicID'])
        for k in range(n):
            df_res['corpus'][k]='Reddit'
            df_res['corpus_year'][k]=dfj['Year'][0]
            df_res['query_source'][k]='Medium'
            df_res['query_year'][k]=years[w]
        store_name="GA_r"+str(dfj['Year'][0])+"m"+str(years[w])+".csv"
        print(store_name)
        df_res.to_csv(store_name)
