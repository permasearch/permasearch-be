from .preprocessor import TextPreprocessor
from .util import get_doc_tokens, extract_doc_id_from_path
from .bsbi import BSBIIndex
from .compression import VBEPostings
from .letor import LETOR
import pickle
import numpy as np

def get_docs(query):

    # instansiasi model untuk test, lalu load model ranker
    letor = LETOR()
    letor.load_model()

    with open("metadata/train_titles.dict", "rb") as f:
        titles_dict = pickle.load(f)
    print(sorted(map(int, titles_dict.keys()))[:100])
    BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')
    
    preprocessor = TextPreprocessor()
    query_tokens = preprocessor.preprocess_text(query)
    
    doc_id_result = []
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=100):
        doc_id_result.append(doc)

    if len(doc_id_result) == 0:
        return doc_id_result
    # membuat matrix fitur
    X_test = []
    for doc_path in doc_id_result:
        doc_tokens = get_doc_tokens(doc_path, preprocessor)
        feature = letor.create_features(query_tokens, doc_tokens)
        X_test.append(feature)
    
    X_test = np.array(X_test)
    
    # lakukan prediksi
    y_pred = letor.predict(X_test)
    
    # mapping doc_id --> score
    path_with_score = {}
    for i in range(len(doc_id_result)):
        path_with_score[doc_id_result[i]] = y_pred[i]
    
    # sort dari score tertinggi
    sorted_dict = dict(sorted(path_with_score.items(), key=lambda item: item[1], reverse=True))
    # mencetak hasil reranking dengan LETOR
    result = []
    for (path, score) in list(sorted_dict.items()):
        data = dict()
        with open(path,'r', encoding='utf-8') as file:
            data['text'] = file.read()
        id = extract_doc_id_from_path(path)
        data['id'] = int(id)
        data['path'] = path.replace('\\', '/')
        data['title'] = titles_dict[id]
        result.append(data)
    return result