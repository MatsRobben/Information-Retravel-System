# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:46:19 2023

@author: Raul
"""

import pandas as pd

import random

qrel = pd.DataFrame(pd.read_table("Documents\Information-Retravel-System\MSMARCO\qrels.dev.tsv",delimiter='\t',header=None))

qrel.columns = ['Topic', 'Iteration', 'Document#', 'Relevancy']

queries = pd.DataFrame(pd.read_table("Documents\Information-Retravel-System\MSMARCO\queries.dev.tsv",delimiter='\t',header=None))

queries.columns = ['qid', 'query']

collection = pd.DataFrame(pd.read_table("Documents\Information-Retravel-System\MSMARCO\collection.tsv",delimiter='\t',header=None))

collection.columns = ['pid', 'passage']

percentage = 99.9

num_rows = int(len(collection) * percentage / 100)

indices = random.sample(range(len(collection)), num_rows)

deleted_values_collection = collection.iloc[indices]['pid'].tolist()

collection = collection.drop(indices).reset_index(drop=True)

deleted_values_qrel = qrel[qrel['Document#'].isin(deleted_values_collection)]['Topic'].tolist()

qrel = qrel[~qrel['Document#'].isin(deleted_values_collection)].reset_index(drop=True)

queries = queries[~queries['qid'].isin(deleted_values_qrel)].reset_index(drop=True)

values_to_keep_queries = qrel['Topic'].tolist()

queries = queries[queries['qid'].isin(values_to_keep_queries)].reset_index(drop=True)

print(qrel)

print(queries)

print(collection)

collection.to_csv('Documents\collection_small.csv', sep=',', index=False)
queries.to_csv('Documents\queries_small.csv', sep=',', index=False)
qrel.to_csv('Documents\qrel_small.csv', sep=',', index=False)