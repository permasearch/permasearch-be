import pickle
import numpy as np
from tqdm import tqdm

from lightgbm import LGBMRanker
from scipy.spatial.distance import cosine

from .lsa import LSA
from .bsbi import BSBIIndex
from .compression import VBEPostings
from .extractor import Extractor

VAL_FILES = {"queries": "val_queries.txt",
                "qrels": "val_qrels.txt"}

TRAIN_FILES = {"docs": "train_docs.txt",
               "queries": "train_queries.txt",
               "qrels": "train_qrels.txt"}

class LETOR:
    # Abstraksi untuk LETOR
    # Terdapat mode untuk LETOR, apakah class ini akan dipanggil untuk train
    # atau test
    def __init__(self, mode="train"):
        self.bsbi = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')
        self.mode = mode
        self.lgbm_model = None
        self.lsa = LSA()
        self.lsa.load_model()

    def create_features(self, query, doc):
        # Method untuk membuat feature vector
        # ambil embedding query dan vector menggunakan LSA model
        vector_query = self.lsa.get_vector_representation(query)
        vector_doc = self.lsa.get_vector_representation(doc)

        # compute jaccard similarity
        set_query = set(query)
        set_doc = set(doc)
        jaccard_sim = len(set_query & set_doc) / len(set_query | set_doc)

        # compute cosine similaritu
        cosine_sim = cosine(vector_query, vector_doc)
        
        # return vector yang kolom-kolomnya menjadi fitur
        return vector_query + vector_doc + [jaccard_sim] + [cosine_sim]

    def generate_ready_to_train_dataset(self):
        self.train_extractor = Extractor(mode="train", files=TRAIN_FILES)
        self.documents = self.train_extractor.read_metadata_docs()
        self.datasets, self.group_qid_count = self.train_extractor.read_metadata_dataset_group_qid_count()
        val_extractor = Extractor(mode="val", files=VAL_FILES)
        self.val_datasets, self.val_group_qid_count = val_extractor.read_metadata_dataset_group_qid_count()

        # Method ini berfungsi untuk generate train dataset
        X = []
        y = []

        for (query, doc, rel) in tqdm(self.datasets, total=len(self.datasets)):
            X.append(self.create_features(query, doc))
            y.append(rel)
        
        return (np.array(X), np.array(y))
    
    def generate_validation_dataset(self):
        # Method ini berfungsi untuk generate validation dataset
        X_val = []
        y_val = []

        for (query, doc, rel) in tqdm(self.val_datasets, total=len(self.val_datasets)):
            X_val.append(self.create_features(query, doc))
            y_val.append(rel)
        
        return (np.array(X_val), np.array(y_val))

    def create_and_train_model(self, write=True):
        self.lsa.generate_model(self.documents)
        # Method ini berfungsi untuk membuat dan train model LGBMRanker
        # dan menulisnya ke metadata bila diperlukan
        model = LGBMRanker(
            objective="lambdarank",
            boosting_type="gbdt",
            n_estimators=100,
            importance_type="gain",
            metric='ndcg',
            num_leaves=40,
            learning_rate=0.02,
            max_depth=-1,
            random_state=42
        )

        # generate train dan validation dataset
        X_train, y_train = self.generate_ready_to_train_dataset()
        X_val, y_val = self.generate_validation_dataset()

        # sambil di-fit, coba evaluasi validation set
        # menggunakan metrik ndcg@5, ndcg@10, ndcg@20
        # sumber: https://tamaracucumides.medium.com/learning-to-rank-with-lightgbm-code-example-in-python-843bd7b44574
        model.fit(X_train, y_train, group=self.group_qid_count,
                  eval_set=[(X_val, y_val)],
                  eval_group=[self.val_group_qid_count],
                  eval_at=[5, 10, 20],
                  eval_names=["validation"])
        
        # set model ke atribut
        self.lgbm_model = model

        if write: self.__write_model()
    
    def __write_model(self):
        # Method ini untuk menulis model ke metadata
        with open(f"model/lgbm_model", "wb") as f:
            pickle.dump(self.lgbm_model, f)
    
    def load_model(self):
        # Method ini untuk membaca model dari metadata
        # lalu assign ke atribut lgbm_model
        with open(f"model/lgbm_model", "rb") as f:
            self.lgbm_model = pickle.load(f)
    
    def get_model(self):
        # Method ini berfungsi untuk mengembalikan model jika diperlukan
        return self.lgbm_model
    
    def predict(self, X):
        # Melakukan prediksi
        return self.lgbm_model.predict(X)

    