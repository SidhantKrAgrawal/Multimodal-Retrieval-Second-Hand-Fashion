from annoy import AnnoyIndex
import numpy as np


class AnnoySearch:
    def __init__(self, dim, metric="angular"):
        self.dim = dim
        self.metric = metric
        self.index = AnnoyIndex(dim, metric)
        self.built = False

    def build(self, embeddings, n_trees=50):
        for i, vec in enumerate(embeddings):
            self.index.add_item(i, vec.tolist())

        self.index.build(n_trees)
        self.built = True

    def search(self, query, k=10):
        assert self.built, "Index not built"

        idxs, dists = self.index.get_nns_by_vector(
            query[0].tolist(), k, include_distances=True
        )

        return np.asarray([idxs]), np.asarray([dists])
