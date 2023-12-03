import pickle
import random
from tqdm import tqdm
from .preprocessor import TextPreprocessor
from .util import get_mapping_doc_id, get_doc_tokens

TRAIN_FILES = {"docs": "train_docs.txt",
               "queries": "train_queries.txt",
               "qrels": "train_qrels.txt",
               "titles": "docs_title.txt"}

VAL_FILES = {"queries": "val_queries.txt",
                "qrels": "val_qrels.txt"}

class Extractor:
    # Class ini adalah abstraksi untuk mengekstrak semua file
    # yang terdapat pada qrels-folder, lalu dapat menyimpan
    # metadata hasil ekstraksinya ke dalam folder metadata
    def __init__(self, root="qrels-folder", index="metadata",
                 files=TRAIN_FILES, mode="train"):
        self.root = root
        self.index = index
        self.mode = mode

        # instansiasi text preprocessor
        self.preprocessor = TextPreprocessor()
        
        if self.mode == "train":
            self.docs_file = files["docs"]
            self.docs_dict = {}

            self.titles_file = files["titles"]
            self.titles_dict = {}

        self.queries_file = files["queries"]
        self.queries_dict = {}

        self.qrels_file = files["qrels"]
        self.qrels_dict = {}  

        # mengambil mapping doc_id --> doc_path
        self.doc_path_mapping = get_mapping_doc_id("write", self.index)           

        # untuk group pada ranker, jumlah dokumen yg berpasangan dgn query tertentu
        self.group_qid_count = []
        self.dataset = []

    def extract_docs(self, write=True):
        # Method ini berfungsi untuk mengekstrak file docs
        if self.mode != "train":
            raise RuntimeError("There is no docs for test or val data")
        
        with open(f"{self.root}/{self.docs_file}", 'r', encoding="utf8") as docs_file:
            total = len(docs_file.readlines())
        
        with open(f"{self.root}/{self.docs_file}", 'r', encoding="utf8") as docs_file:
            # looping tiap line, preprocess lalu simpan ke docs_dict
            # docs_dict berisi mapping doc_id --> doc_tokens
            for line in tqdm(docs_file, total=total):
                doc_id, content = line.split()[0], " ".join(line.split()[1:])
                self.docs_dict[doc_id] = self.preprocessor.preprocess_text(content)
        
        if write: self.__write_metadata_docs()

    def __write_metadata_docs(self):
        # Method ini berfungsi untuk menulis docs_dict ke metadata
        with open(f"{self.index}/{self.mode}_docs.dict", "wb") as f:
            pickle.dump(self.docs_dict, f)
    
    def read_metadata_docs(self):
        # Method ini berfungsi untuk membaca docs_dict dari metadata, lalu
        # di-load untuk diassign ke atribut docs_dict dan mengembalikannya
        with open(f"{self.index}/{self.mode}_docs.dict", "rb") as f:
            self.docs_dict = pickle.load(f)
        return self.docs_dict
    
    def extract_titles(self, write=True):
        if self.mode != "train":
            raise RuntimeError("There is no docs for test or val data")
        
        with open(f"{self.root}/{self.titles_file}", 'r', encoding="utf8") as titles_file:
            total = len(titles_file.readlines())
        
        with open(f"{self.root}/{self.titles_file}", 'r', encoding="utf8") as titles_file:
            for line in tqdm(titles_file, total=total):
                doc_id, title = line.split()[0], " ".join(line.split()[1:])
                self.titles_dict[doc_id] = title
        
        if write: self.__write_metadata_titles()

    def __write_metadata_titles(self):
        with open(f"{self.index}/{self.mode}_titles.dict", "wb") as f:
            pickle.dump(self.titles_dict, f)
    
    def read_metadata_titles(self):
        with open(f"{self.index}/{self.mode}_titles.dict", "rb") as f:
            self.titles_dict = pickle.load(f)
        return self.titles_dict
    
    def extract_queries(self, write=True):
        # Method ini berfungsi untuk mengekstrak file yang berisi queries
        with open(f"{self.root}/{self.queries_file}", 'r', encoding="utf-8") as queries_file:
            total = len(queries_file.readlines())

        with open(f"{self.root}/{self.queries_file}", 'r', encoding="utf-8") as queries_file:
            # looping tiap line, preprocess lalu simpan ke queries_dict
            # queries_dict berisi mapping query_id --> query_tokens
            for line in tqdm(queries_file, total=total):
                query_id, content = line.split()[0], " ".join(line.split()[1:])
                self.queries_dict[query_id] = self.preprocessor.preprocess_text(content)
        
        if write: self.__write_metadata_queries()
    
    def __write_metadata_queries(self):
        # Method ini berfungsi untuk menulis queries_dict ke metadata
        with open(f"{self.index}/{self.mode}_queries.dict", "wb") as f:
            pickle.dump(self.queries_dict, f)
    
    def read_metadata_queries(self):
        # Method ini berfungsi untuk membaca queries_dict dari metadata, lalu
        # di-load untuk diassign ke atribut queries_dict dan mengembalikannya
        with open(f"{self.index}/{self.mode}_queries.dict", "rb") as f:
            self.queries_dict = pickle.load(f)
        return self.queries_dict
    
    def extract_qrels(self, write=True):
        # Method ini generate qrels_dict dengan format di bawah
        # {query_id: [(doc_id1, rel1), (doc_id2, rel2), ...]}
        if self.mode == "train":
            self.read_metadata_docs()
        self.read_metadata_queries()

        with open(f"{self.root}/{self.qrels_file}", 'r', encoding='utf-8') as qrels_file:
            total = len(qrels_file.readlines())

        with open(f"{self.root}/{self.qrels_file}", 'r', encoding='utf-8') as qrels_file:
            if self.mode == "train":
                for line in tqdm(qrels_file, total=total):
                    query_id, doc_id, rel = line.split()
                    if (query_id in self.queries_dict) and (
                        doc_id in self.docs_dict):
                        if query_id not in self.qrels_dict:
                            self.qrels_dict[query_id] = []
                        self.qrels_dict[query_id].append((doc_id, int(rel)))
            elif self.mode == "val":
                for line in tqdm(qrels_file, total=total):
                    query_id, doc_id, rel = line.split()
                    # validation tidak berdasarkan docs_dict
                    if query_id in self.queries_dict:
                        if query_id not in self.qrels_dict:
                            self.qrels_dict[query_id] = []
                        self.qrels_dict[query_id].append((doc_id, int(rel)))
        
        if write: self.__write_metadata_qrels()
    
    def __write_metadata_qrels(self):
        # Method ini berfungsi untuk menulis raw_queries_dict ke metadata
        with open(f"{self.index}/{self.mode}_qrels.dict", "wb") as f:
            pickle.dump(self.qrels_dict, f)
    
    def read_metadata_qrels(self):
        # Method ini berfungsi untuk membaca qrels_dict dari metadata, lalu
        # di-load untuk diassign ke atribut qrels_dict dan mengembalikannya
        with open(f"{self.index}/{self.mode}_qrels.dict", "rb") as f:
            self.qrels_dict = pickle.load(f)
        return self.qrels_dict

    def generate_dataset(self, NUM_NEGATIVES=1, write=True):
        # Method ini berfungsi untuk generate dataset yang akan
        # digunakan saat training dan prediction
        if self.mode == 'train':
            self.read_metadata_docs()
        self.read_metadata_queries()
        self.read_metadata_qrels()
        
        # penambahan docs yang tidak relevan hanya untuk training
        if self.mode != 'train': NUM_NEGATIVES = 0

        for query_id in tqdm(self.qrels_dict, total=len(self.qrels_dict)):
            # qrels_dict berisi list dokumen yang relevan beserta relevansinya
            # [(doc_id1, rel1), (doc_id2, rel2), ...]
            docs_rels = self.qrels_dict[query_id]
            # ini untuk pengelompokkan query untuk training
            self.group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
           
            if self.mode == 'train':
                for doc_id, rel in docs_rels:
                    # dataset: (query_embedding, doc_embedding, doc_id, raw_query, rel)
                    # doc_id dan raw_query digunakan untuk retrieve score tfidf dan bm25
                    self.dataset.append((self.queries_dict[query_id],
                                    self.docs_dict[doc_id], 
                                    rel))
            else:
                for doc_id, rel in docs_rels:
                    doc_tokens = get_doc_tokens(self.doc_path_mapping[doc_id], self.preprocessor)
                    self.dataset.append((self.queries_dict[query_id], 
                                        doc_tokens, 
                                        rel))
                    
            for _ in range(NUM_NEGATIVES):
                # masukkan contoh yang tidak relevan (untuk training)
                rand_doc_id = random.choice(list(self.docs_dict.keys()))
                self.dataset.append((self.queries_dict[query_id],
                                self.docs_dict[rand_doc_id],
                                0))
        
        if write: self.__write_metadata_dataset_group_qid_count()
        
    def __write_metadata_dataset_group_qid_count(self):
        # Method ini berfungsi untuk menulis dataset dan group_qid_count ke metadata
        with open(f"{self.index}/{self.mode}_dataset.list", "wb") as f:
            pickle.dump(self.dataset, f)
        with open(f"{self.index}/{self.mode}_group_qid_count.list", "wb") as f:
            pickle.dump(self.group_qid_count, f)

    def read_metadata_dataset_group_qid_count(self):
        # Method ini berfungsi untuk membaca dataset dan group_qid_count dari metadata, lalu
        # di-load untuk diassign ke atribut dataset dan group_qid_count dan mengembalikannya
        with open(f"{self.index}/{self.mode}_dataset.list", "rb") as f:
            self.dataset = pickle.load(f)
        with open(f"{self.index}/{self.mode}_group_qid_count.list", "rb") as f:
            self.group_qid_count = pickle.load(f)
        return (self.dataset, self.group_qid_count)