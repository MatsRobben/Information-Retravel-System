# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:27:21 2023

@author: Raul
"""

from sklearn.metrics import ndcg_score
import pandas as pd

def calculate_ndcg(qrel_file, results_file):

    # Load results file
    results_df = pd.read_csv(results_file, sep=',', header=None, names=['query', 'relevancy', 'Document#', 'rank', 'score', 'run'])
    
    print(results_df.head(5))

    # Group by query and calculate NDCG score for each query
    ndcg_scores = []
    for query, group in results_df.groupby('query'):
        y_true = group['relevancy'].values
        y_score = group['score'].values
        ndcg = ndcg_score([y_true], [y_score])
        ndcg_scores.append(ndcg)
        
    print(ndcg_scores)

    return ndcg_scores

calculate_ndcg(r"C:\Users\Raul\Documents\Information-Retravel-System\MSMARCO_SMALL\qrel_small.csv", r"C:\Users\Raul\Documents\Information-Retravel-System\BM25_evalcsv.csv")