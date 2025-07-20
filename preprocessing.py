import string
from bs4 import BeautifulSoup
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

nltk.download('stopwords')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = TreebankWordTokenizer()
        self.stemmer = PorterStemmer()
        self.translator = str.maketrans('', '', string.punctuation)

    def clean_text(self, text):
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        text = text.translate(self.translator)
        tokens = self.tokenizer.tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        tokens = [t for t in tokens if not t.isnumeric()]
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self.clean_text)
