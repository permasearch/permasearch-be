import pickle
from gensim.models import LsiModel
from gensim.corpora import Dictionary

class LSA:
    # Class ini berfungsi untuk abstraksi LSA
    def __init__(self, num_latent_topics=100):
        self.num_latent_topics = num_latent_topics
        self.model = None
        self.dictionary = Dictionary()
    
    def generate_model(self, documents):
        # documents berisi dictionary {doc_id1: [token1, token2, ...], ...}
        # convert list of tokens menjadi bag of words
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True)
                      for doc in documents.values()]
        # buat model dari corpus
        model = LsiModel(bow_corpus, num_topics=self.num_latent_topics,
                         random_seed=42)
        # set model
        with open(f"model/lsa_model", "wb") as f:
            pickle.dump(model, f)
    
    def load_model(self):
        # Method ini untuk membaca model dari metadata
        # lalu assign ke atribut lgbm_model
        with open(f"model/lsa_model", "rb") as f:
            self.model = pickle.load(f)

    def get_vector_representation(self, tokens):
        # create word embedding dari LSA model untuk tokens
        representation = [topic_value 
                          for (_, topic_value) 
                          in self.model[self.dictionary.doc2bow(tokens)]]
        
        if len(representation) == self.num_latent_topics:
            return representation
        else:
            return [0.] * self.num_latent_topics