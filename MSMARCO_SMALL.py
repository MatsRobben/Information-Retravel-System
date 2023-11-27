# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:46:19 2023

@author: Raul
"""

# basis imports needed
import pandas as pd

import random

# Get initial qrel file and make it a dataframe with corresponding column names
qrel = pd.DataFrame(pd.read_table("data\qrels.dev.tsv",delimiter='\t',header=None))

qrel.columns = ['Topic', 'Iteration', 'Document#', 'Relevancy']

# Get intitial queries and make a dataframe out of it with corresponding column names
queries = pd.DataFrame(pd.read_table("data\queries.dev.tsv",delimiter='\t',header=None))

queries.columns = ['qid', 'query']

# Get intitial queries and make a dataframe out of it with corresponding column names
collection = pd.DataFrame(pd.read_table("data\collection.tsv",delimiter='\t',header=None))

collection.columns = ['pid', 'passage']

# Percentage to get rid of
percentage = 99.9

# Getting the indices of the rows to get rid of
num_rows = int(len(collection) * percentage / 100)

indices = random.sample(range(len(collection)), num_rows)

#remember the deleted values of the pid for the deleted rows from the corpus
deleted_values_collection = collection.iloc[indices]['pid'].tolist()

#drop the rows
collection = collection.drop(indices).reset_index(drop=True)

# get topics/queries that are going to be deleted from the qrel + delete qrel rows by using corresponding pid (if passage is not there in the corpus, we won't use that row either)
deleted_values_qrel = qrel[qrel['Document#'].isin(deleted_values_collection)]['Topic'].tolist()

qrel = qrel[~qrel['Document#'].isin(deleted_values_collection)].reset_index(drop=True)

# get rid of the queries if the qid is not in the topic of the qrel file anymore
queries = queries[~queries['qid'].isin(deleted_values_qrel)].reset_index(drop=True)

# all qrel topics/queries that are left
values_to_keep_queries = qrel['Topic'].tolist()

# if there are no passages with relevancy 1 in the qrel file (topic not there) we get rid of those queries/topics from the queries file.
queries = queries[queries['qid'].isin(values_to_keep_queries)].reset_index(drop=True)

print(qrel)

print(queries)

print(collection)

# to csv
collection.to_csv('MSMARCO_SMALL\collection_small.csv', sep=',', index=False)
queries.to_csv('MSMARCO_SMALL\queries_small.csv', sep=',', index=False)
qrel.to_csv('MSMARCO_SMALL\qrel_small.csv', sep=',', index=False)