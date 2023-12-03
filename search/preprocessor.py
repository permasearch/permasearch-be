from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

class TextPreprocessor:
    # Class ini sebagai abstraksi untuk melakukan preprocessing text
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, doc):
        # tokenization
        tokenized = re.findall(r"\w+", doc)
        tokenized = ' '.join(tokenized)

        tokens = [token.lower() for token in tokenized.split() 
                  if token.lower() not in self.stop_words]
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]

        return stemmed_tokens
