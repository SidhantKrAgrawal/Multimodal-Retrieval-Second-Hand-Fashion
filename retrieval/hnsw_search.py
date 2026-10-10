import hnswlib
import numpy as np

class HNSWSearch:
    def __init__(self, dim, space="cosine"):
        self.dim = dim
        self.index = hnswlib.Index(space=space, dim=dim)

    def build(self, embeddings, ef_construction=200, M=16, ef_search=50):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        num_elements = embeddings.shape[0]

        self.index.init_index(
            max_elements=num_elements,
            ef_construction=ef_construction,
            M=M
        )
        self.index.add_items(embeddings, np.arange(num_elements))
        self.index.set_ef(ef_search)

    def search(self, query, k=10):
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        labels, distances = self.index.knn_query(query, k=k)
        return labels, distances

