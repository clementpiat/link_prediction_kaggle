import pandas as pd
from collections import Counter
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from edge_metrics import all_metrics
from word2vec import load_data
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords

class TFIDF:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english')) 
        self.wnl = WordNetLemmatizer()
        self.token_stop = self.tokenizer(' '.join(self.stop_words))

    def tokenizer(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]



class Embeddings:
    def __init__(self):
        self.node_information = pd.read_csv('node_information.csv', header=None, names=['node', 'year', 'title', 'authors', 'journal', 'abstract']).fillna('')
        self.node_to_index = {node: i for i, node in enumerate(self.node_information.node.values)}




