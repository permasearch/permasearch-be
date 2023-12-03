import os
import pickle

from tqdm import tqdm


class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # Jika s ada pada IdMap, kembalikan langsung value integer yang bersesuaian
        # Jika s tidak ada pada IdMap, kembalikan value integer yang baru di-assign, yakni
        # value panjang IdMap tersebut
        value_id = self.str_to_id.get(s, len(self))
        
        # Jika value_id sama dengan panjang IdMap, maka itu pasti value yang baru di-assign
        # Sehingga perlu ditambahkan ke str_to_id dan id_to_str
        if value_id == len(self):
            self.str_to_id[s] = value_id
            self.id_to_str.append(s)

        return value_id

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        return self.id_to_str[i]

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        if type(key) is str:
            return self.__get_id(key)
        else:
            return self.__get_str(key)


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparable, int)]
        Penggabungan yang sudah terurut
    """
    # Inisialisasi pointer dan list kosong
    pointer_tfs1 = 0
    pointer_tfs2 = 0
    result = []
    # Mirip seperti algoritma untuk UNION
    # Referensi: https://scele.cs.ui.ac.id/pluginfile.php/193210/mod_resource/content/1/OR_ANDNOT.txt
    while pointer_tfs1 != len(posts_tfs1) and pointer_tfs2 != len(posts_tfs2):
        if posts_tfs1[pointer_tfs1][0] == posts_tfs2[pointer_tfs2][0]:
            # Membuat tuple yang berisi (docID, accumulatedTF)
            entry = (posts_tfs1[pointer_tfs1][0], 
                     posts_tfs1[pointer_tfs1][1] + posts_tfs2[pointer_tfs2][1])
            result.append(entry)
            pointer_tfs1 += 1
            pointer_tfs2 += 1
        elif posts_tfs1[pointer_tfs1][0] < posts_tfs2[pointer_tfs2][0]:
            # Menambahkan entry untuk docID pada posts_tfs1 yang lebih kecil
            result.append(posts_tfs1[pointer_tfs1])
            pointer_tfs1 += 1
        else:
            # Menambahkan entry untuk docID pada posts_tfs2 yang lebih besar
            result.append(posts_tfs2[pointer_tfs2])
            pointer_tfs2 += 1
        
    # Sisa entries (jika ada) tinggal di-append ke result saja
    while pointer_tfs1 != len(posts_tfs1):
        result.append(posts_tfs1[pointer_tfs1])
        pointer_tfs1 += 1
    while pointer_tfs2 != len(posts_tfs2):
        result.append(posts_tfs2[pointer_tfs2])
        pointer_tfs2 += 1
    
    return result

# mode: "write" or "read", kalau read asumsi metadata sudah ada
def get_mapping_doc_id(mode="write", index="index"):
    # Method ini berfungsi untuk mengambil mapping doc_id --> path
    doc_id_to_path = {}
    if mode == "write":
        total = len(os.listdir("collections"))
        for folder in tqdm(os.listdir("collections"), total=total):
            # Looping untuk memasukkan doc_id --> path ke dictionary
            for filename in os.listdir(os.path.join("collections", folder)):
                path = str(os.path.join("collections", folder, filename))
                doc_id = filename.split(".")[0]
                if doc_id in doc_id_to_path:
                    raise KeyError("Document ID is already in dictionary")
                doc_id_to_path[doc_id] = path
        
        # Simpan metadata
        with open(f"{index}/doc_path_mapping.dict", "wb") as f:
            pickle.dump(doc_id_to_path, f)
    
    else:
        # Jika read, asumsi sudah ada, tinggal baca metadata
        with open(f"{index}/doc_path_mapping.dict", "rb") as f:
            doc_id_to_path = pickle.load(f)
    
    return doc_id_to_path

def extract_doc_id_from_path(path):
    return (path.split("\\")[-1]).split(".")[0]

def get_doc_tokens(path, preprocessor):
    with open(path.replace('\\', '/'), 'r', encoding='utf-8') as file:
        content = file.read()
    return preprocessor.preprocess_text(content)