from sentence_transformers import util


class SearchEngine:

    def __init__(self, db):
        self.db = db

    def _run_query(self, query_text):
        _res = self.db.vector_search(query_text)
        print(_res)



