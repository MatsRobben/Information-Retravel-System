from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import random
import pickle
import gzip
import math
from collections import defaultdict


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(dict)
        self.doc_lengths = {}
        self.avgdl = 0
        self.idf = {}

    def get_index(self):
        return self.index

    def get_doc_lengths(self):
        return self.doc_lengths

    def get_avgdl(self):
        return self.avgdl

    def get_idf(self):
        return self.idf

    def load_file(self, file_name):
        file_extension = file_name.split('.')[-1].lower()

        if file_extension == 'csv':
            self.docs = pd.read_csv(file_name)
        elif file_extension == 'tsv':
            self.docs = pd.read_csv(file_name, delimiter='\t',header=None)
            self.docs.columns = ['pid', 'passage']
        else:
            raise ValueError("Unsupported file format. Supported formats: CSV (.csv) and TSV (.tsv)")

    def build_index(self, file_name: str):
        self.load_file(file_name=file_name)

        self.docs['passage'] = self.docs['passage'].apply(preprocess_text)

        total_tokens = 0
        for index, row in self.docs.iterrows():
            doc_id, tokens = row['pid'], row['passage']
            total_tokens += len(tokens)
            for term in tokens:
                self.index[term][doc_id] = self.index[term].get(doc_id, 0) + 1

            self.doc_lengths[doc_id] = len(tokens)

        self.avgdl = total_tokens / len(self.docs)
        self.compute_idf()

    def compute_idf(self):
        total_docs = len(self.docs)
        for term in self.index:
            doc_freq = len(self.index[term])
            self.idf[term] = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def save_index(self, file_name: str):
        with gzip.open(file_name, 'wb', compresslevel=5) as file:
            pickle.dump({'index': self.index, 'doc_lengths': self.doc_lengths, 'avgdl': self.avgdl, 'idf': self.idf}, file)

    def load_index(self, file_name: str):
        with gzip.open(file_name, 'rb') as file:
            data = pickle.load(file)
            self.index = data['index']
            self.doc_lengths = data['doc_lengths']
            self.avgdl = data['avgdl']
            self.idf = data['idf']


def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
        return tokens