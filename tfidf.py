from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

from utils import load_node_information

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english')) 

# Interface lemma tokenizer from nltk with sklearn
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

# Lemmatize the stop words
tokenizer=LemmaTokenizer()
token_stop = tokenizer(' '.join(stop_words))

vectorizer = TfidfVectorizer(strip_accents='unicode',
                        lowercase=True, analyzer='word', token_pattern=r'\w+',
                        use_idf=True, smooth_idf=True, sublinear_tf=True, 
                        stop_words=token_stop, tokenizer=tokenizer)

svd = TruncatedSVD(n_components=10, n_iter=10, random_state=42)

if __name__ == '__main__':
    node_information = load_node_information()

    X_tfidf = vectorizer.fit_transform(node_information.abstract.values)
    X_svd = svd.fit_transform(X_tfidf)

    with open(f'emb_nodes/tfidf.npy', 'wb') as f:
        np.save(f, X_svd)

