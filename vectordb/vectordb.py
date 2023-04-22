import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import torch
from sentence_transformers import SentenceTransformer, util


class VectorDB:

    def __init__(self, model_name = 'all-MiniLM-L12-v2', encode_batch_size = 64):
        self.model_name = model_name
        self.batch_size = encode_batch_size
        self.fcontents = []
        self.db = []
        self.json_out_file_name = os.getcwd()+'/vect_db.json'
        self.db_path = os.getcwd()+'/db.vectordb'

        self.model = SentenceTransformer(model_name)

    def generate_db(self, save_json = False):
        """
        generates embeddings of read file contents
        and saves them as .vectordb
        """
        _embeddings = self.model.encode([fc['content'] for fc in self.fcontents], convert_to_tensor=True, show_progress_bar=True, batch_size=self.batch_size)
        self.db = {
            'fcontents': self.fcontents,
            'embeddings': _embeddings,
            'model_name': self.model_name
             }
        self.pickle_db()
        if save_json:
            self.dumb_db_as_json()

    def process_folder(self, dir_path, ext_filter):
        if not self.isdir(dir_path):
            raise NotADirectoryError("path specified is not a directory.")
        _files = self.get_filepaths(dir_path, ext_filter)
        self.fcontents.extend(self.read_files(_files))


    def read_files(self, files):
        file_content_map = []
        for file in files:
            with open(file, 'r') as f:
                #file_content = f.read()
                for idx,line in enumerate(f.read().splitlines()):
                    if len(line) > 0:
                        file_content_map.append( {'file-name': file, 'line_number': idx+1, 'content': line.rstrip()})
        return file_content_map

    def get_filepaths(self, directory, extensions: Tuple[str]):
        """
        explore a directory and return files with extension in extensions
        """
        file_paths = []

        for root, directories, files in os.walk(directory):
            for filename in files:
                if filename.endswith(extensions):
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)

        return file_paths

    def pickle_db(self):
        with gzip.open(self.db_path, 'w') as f:
            f.write(pickle.dumps(self.db))

    def load_db(self, path = None):
        if path is None:
            path = self.db_path
        _db = {}
        with gzip.open(path, 'r') as f:
            _db = pickle.loads(f.read())
            if _db.get('model_name') != self.model_name:
                print('db was generated using a different model, regenerating db... \n please try again')
                self.generate_db()
            else :
                self.db =_db

    def get_embedding(self, q_str):
        return self.model.encode(q_str, convert_to_tensor=True)

    def vector_search(self, q_str, limit):
        q_emb = self.get_embedding(q_str)
        cosine_sim = util.cos_sim(q_emb, self.db['embeddings'])[0]
        _top_results = torch.topk(cosine_sim, k=min(limit, len(cosine_sim)), sorted=True)
        _res = []
        for score, idx in zip(_top_results[0], _top_results[1]):
            _res.append((score, self.db['fcontents'][idx]))
        return _res

    def isdir(self, directory):
        """ creates directory if it doesn't exist """
        directory = Path(directory)
        if not directory.is_dir():
            return False
        return True


    def dumb_db_as_json(self):
        """ write json """
        file = Path(self.json_out_file_name)
        _db =  self.db.copy()
        _db['metadata'] = 'viz only'
        _db['embeddings'] = [str(e) for e in self.db['embeddings']]
        with file.open('wt') as handle:
            json.dump(_db, handle, indent=4, sort_keys=False)