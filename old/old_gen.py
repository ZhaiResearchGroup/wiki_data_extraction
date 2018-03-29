
# coding: utf-8

# In[1]:


from subprocess import call
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import itertools
from multiprocessing import Pool

import metapy
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


# In[2]:


NUM_PROCESSES=8


# In[3]:


df = pd.read_csv('wiki_old_input.csv')


# Set up error logging

# In[4]:


formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# modified from https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
def generate_logger(name):
    handler = logging.FileHandler('{}.log'.format(name), mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger

loggers = [generate_logger('search/wiki_{}/wiki_{}'.format(i, i)) for i in range(1, 9)]


# Train vectorizer

# In[5]:


corpus = df['document'].tolist()
len(corpus)


# In[6]:


vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)


# In[7]:


# transform our raw queries using tf idf
def transform_query(raw_query):
    sparse_query = vectorizer.transform([raw_query])
    return sparse_query


# Extract the queries using MeTaPy

# In[8]:


def get_sentences(summary):
    return [sentence.strip() for sentence in summary.split(".")]

def write_doc(document, filename, p_index):
    sentences = [sentence.strip() for sentence in document.split(".")]
    
    # get rid of empty strings: https://stackoverflow.com/questions/3845423/remove-empty-strings-from-a-list-of-strings
    sentences = list(filter(None, sentences))

    # write document to file
    with open(filename, 'w+') as doc_file:
        for sentence in sentences:
            doc_file.write("{}\n".format(sentence))
    
    # write metadata
    with open('search/wiki_{}/metadata.dat'.format(p_index), 'w+') as meta_file:
        for i in range(len(sentences)):
            meta_file.write("SEN{}\n".format(i))

def remove_old_idx(p_index):
    call(["rm", "-r", "search/idx_{}".format(p_index)])
    
def get_stringified_list(idx, search_results):
    return [idx.metadata(doc_id).get('content') for (doc_id, score) in search_results]

def search(document, summary, p_index):
    write_doc(document, 'search/wiki_{}/wiki_{}.dat'.format(p_index, p_index), p_index)
    remove_old_idx(p_index)
    
    idx = metapy.index.make_inverted_index('search/config_{}.toml'.format(p_index))
    
    ranker = metapy.index.OkapiBM25()
    
    query = metapy.index.Document()
    query.content(summary)
    
    search_results = ranker.score(idx, query, num_results=5)
    return get_stringified_list(idx, search_results)


# Generate raw and transform queries

# In[9]:


def generate_queries(args):
    p_df, p_index = args
    
    logger = loggers[p_index - 1]

    data = {"title": [], "raw_query": [], "sentence_summary": [], "document": []}
    
    queries = []

    documents_generated = 0
    queries_generated = 0

    total_documents = p_df.shape[0]

    for row in p_df.iterrows():
        title = row[1]['title']
        summary = row[1]['summary']
        document = row[1]['document']
        
        logger.debug(title)
        
        if (not document.strip()) or (not summary.strip()):
            continue
        
        sentences = get_sentences(summary)
        sentences = list(filter(None, sentences))
        
        # only take first 3 sentences
        sentences = sentences[:3]

        for sentence in sentences:

            # extract query
            raw_queries = search(document, sentence, p_index)
            
            for raw_query in raw_queries:
                query = transform_query(raw_query)

                # add query info to data
                data["raw_query"].append(raw_query)
                data["sentence_summary"].append(sentence)
                data["title"].append(title)
                data["document"].append(document)

                queries.append(query)
            
                queries_generated += 1
        
        documents_generated += 1
        
        if documents_generated % 20 == 0:
            print("Process {}: Generated {} queries for {} documents, {:.4f}% complete".format(p_index, queries_generated, documents_generated, documents_generated/total_documents * 100))
    
    print("Process {}: Finished generating queries".format(p_index))

    return pd.DataFrame(data=data), sp.sparse.vstack(queries)


# In[10]:


def store_queries_data(df_queries, queries):
    # reorganize dataframe
    df_final = df_queries[['title', 'raw_query', 'sentence_summary', 'document']]
    df_final = df_final.reindex(df_final.index.rename('query_index'))
    df_final.index = df_final.index.astype(int)
    
    # store queries matrix
    sp.sparse.save_npz("out/queries_matrix.npz", queries)

    # store queries csv
    df_final.to_csv("out/wiki_queries.csv")


# In[41]:


def generate_queries_multiprocess(df, num_processes=NUM_PROCESSES):  
    pool = Pool(processes=num_processes)
    
    p_dfs = np.array_split(df, num_processes)
    
    args_by_process = [(p_dfs[i], i+1) for i in range(len(p_dfs))]
    results = pool.map(generate_queries, args_by_process)

    pool.close()

    df_queries = pd.concat([result[0] for result in results], ignore_index=True)
    queries = sp.sparse.vstack([result[1] for result in results])
    
    return df_queries, queries

def create_queries():
    print("Started generating queries")

    df_queries, queries = generate_queries_multiprocess(df)

    store_queries_data(df_queries, queries)

    print("Finished generating queries")

create_queries()


# In[12]:


# store vectorizer
pickle_out = open("out/vectorizer.pickle","wb")
pickle.dump(vectorizer, pickle_out)
pickle_out.close()


# In[18]:


# out_df = pd.read_csv('out/wiki_queries.csv')


# In[19]:


# out_df


# In[22]:


# row = out_df.iloc[0]


# In[37]:


# def print_info(print_df):
#     for row in print_df.iterrows():
#         print("----- QUERY -----")
#         print(row[1]['raw_query'])
#         print("\n")
#         print("----- SUMMARY -----")
#         print(row[1]['sentence_summary'])
#         print("\n--------------------\n")

# print_info(out_df.head())


# In[39]:


# print_info(out_df.loc[out_df.raw_query.str.len() < .5 * out_df.sentence_summary.str.len(), ['raw_query', 'sentence_summary']])


# In[40]:


# IDEAS:
# - pull stuff from the right-hand side as queries (ex/ "Created by")
# - pull content headers as queries (ex/ "origins")
# - use the title as another query

