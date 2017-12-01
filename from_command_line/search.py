import pymongo
import re
import requests
import pandas as pd
import numpy as np
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import sys

client = pymongo.MongoClient('54.200.30.7', 27016)

def search_term(search_term):
    
    cursor_ml_level_0 = client.machine_learning.level_0.find()
    db_ml_level_0 = list(cursor_ml_level_0)
    df_ml_level_0 = pd.DataFrame(db_ml_level_0)
    
    cursor_bs_level_0 = client.business_software.level_0.find()
    db_bs_level_0 = list(cursor_bs_level_0)
    df_bs_level_0 = pd.DataFrame(db_bs_level_0)
    
    df = pd.concat([df_ml_level_0, df_bs_level_0])
    
    top_5_list = []
    final_df = pd.DataFrame()
    
    tfidf_vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
    initial_terms_encoded = tfidf_vectorizer.fit_transform(df['raw page text'])
    initial_terms_encoded_df = pd.DataFrame(initial_terms_encoded.toarray(), 
                                       index=df['raw page text'], 
                                       columns=tfidf_vectorizer.get_feature_names())
    
    search_terms_encoded = tfidf_vectorizer.transform([search_term])
    search_terms_encoded_df = pd.DataFrame(search_terms_encoded.toarray(), 
                                       index=[search_term], 
                                       columns=tfidf_vectorizer.get_feature_names())
    
    dtm_with_search_term = initial_terms_encoded_df.append(search_terms_encoded_df)
    n_components = 200
    SVD = TruncatedSVD(n_components)
    component_names = ["component_"+str(i+1) for i in range(n_components)]
    
    svd_matrix = SVD.fit_transform(dtm_with_search_term)
    svd_df = pd.DataFrame(svd_matrix, 
                      index=dtm_with_search_term.index, 
                      columns=component_names)
    
    search_term_svd_vector = svd_df.loc[search_terms_encoded_df.index]
    svd_df['cosine_sim'] = cosine_similarity(svd_df, search_term_svd_vector)
    top_5 = svd_df[['cosine_sim']].sort_values('cosine_sim', ascending=False).head(6)
    
    for page in range(1,6):
        top_5_list.append(df[df['raw page text'] == top_5.index[page]]['title'])
    
    final_df['title'] = pd.concat(top_5_list)
    return final_df

q = 0
while q == 0:
    name = raw_input("Search (type 'q' to exit): ")
    if name == 'q':
        q = 1
        pass
    if name != 'q':
        search = search_term(name)
        print(search)