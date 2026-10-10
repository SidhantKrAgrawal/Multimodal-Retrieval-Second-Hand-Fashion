import math
import re
from collections import Counter, defaultdict


def label_to_text(label: dict) -> str:
    """Convert a label dict to a weighted flat text string for BM25/TF-IDF indexing."""
    parts = []

    def _add(value, repeat=1):
        if not value:
            return
        token = str(value).lower().strip()
        if token:
            parts.extend([token] * repeat)

    def _add_list(lst, repeat=1):
        for v in (lst or []):
            _add(v, repeat)

    _add(label.get("category"), repeat=4)
    _add(label.get("type"),     repeat=3)
    _add_list(label.get("colors",  []), repeat=2)
    _add(label.get("season"),   repeat=1)
    _add(label.get("brand"),    repeat=1)
    _add(label.get("pattern"),  repeat=1)
    _add(label.get("material"), repeat=1)
    _add_list(label.get("cut",  []), repeat=1)

    return " ".join(parts)


def _tokenise(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Corpus:
    """BM25 index over label dicts. Indices match the embedding matrix order."""

    def __init__(self, aligned_labels: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.N  = len(aligned_labels)

        self._docs: list[list[str]] = [
            _tokenise(label_to_text(lbl)) for lbl in aligned_labels
        ]
        self._doc_lens = [len(d) for d in self._docs]
        self._avgdl    = sum(self._doc_lens) / max(self.N, 1)

        self._inv_idx: dict[str, dict[int, int]] = defaultdict(dict)
        for idx, tokens in enumerate(self._docs):
            tf = Counter(tokens)
            for term, freq in tf.items():
                self._inv_idx[term][idx] = freq

        self._idf: dict[str, float] = {}
        for term, postings in self._inv_idx.items():
            df = len(postings)
            self._idf[term] = math.log(
                (self.N - df + 0.5) / (df + 0.5) + 1.0
            )

    def _score_query(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        k1, b, avgdl = self.k1, self.b, self._avgdl

        for term in set(query_tokens):
            if term not in self._inv_idx:
                continue
            idf = self._idf[term]
            postings = self._inv_idx[term]
            for doc_idx, tf in postings.items():
                dl    = self._doc_lens[doc_idx]
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc_idx] += idf * (tf * (k1 + 1)) / denom

        return scores

    def search(self, query_idx: int, k: int) -> list[tuple[int, float]]:
        """Return top-k (item_idx, score) pairs, excluding the query item."""
        query_tokens = self._docs[query_idx]
        if not query_tokens:
            return []

        scores = self._score_query(query_tokens)
        scores[query_idx] = -1.0
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(idx, sc) for idx, sc in indexed[:k] if sc > 0]


class TFIDFCorpus:
    """TF-IDF retrieval using scikit-learn. Same .search(query_idx, k) interface as BM25Corpus."""

    def __init__(self, aligned_labels: list):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        self._np = np
        corpus = [label_to_text(lbl) for lbl in aligned_labels]
        self._vec = TfidfVectorizer(
            lowercase=True, token_pattern=r"[a-z0-9]+",
            sublinear_tf=True,
            max_features=None,
        )
        self._matrix = self._vec.fit_transform(corpus)

    def search(self, query_idx: int, k: int) -> list[tuple[int, float]]:
        query_vec = self._matrix[query_idx]
        scores = (self._matrix @ query_vec.T).toarray().flatten()
        scores[query_idx] = -1.0
        top_k_idx = self._np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top_k_idx if scores[i] > 0]
