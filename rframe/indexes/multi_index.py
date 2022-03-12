from collections.abc import Mapping

import toolz

from .base import BaseIndex


class FrozenDict(Mapping):
    """https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be"""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash


def hashable_doc(doc):
    if isinstance(doc, dict):
        doc = {k: hashable_doc(v) for k, v in doc.items()}
        return FrozenDict(doc)
    return doc


def unhashable_doc(doc):
    if isinstance(doc, FrozenDict):
        doc = {k: unhashable_doc(v) for k, v in doc.items()}
        return dict(doc)
    return doc


class MultiIndex(BaseIndex):
    _indexes: list

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        indexes = list(args)

        for i, index in enumerate(indexes):
            if index.name in ["", "index"]:
                index.name = f"index_{i}"
        self._indexes = indexes
        
    @property
    def DOCS_PER_LABEL(self):
        ndocs = 1
        for index in self.indexes:
            ndocs *= index.DOCS_PER_LABEL
        return ndocs

    @property
    def indexes(self):
        return getattr(self, "_indexes", [])[:]

    @property
    def names(self):
        return [index.name for index in self.indexes]

    def validate_label(self, label: dict) -> dict:
        indexes = {index.name: index for index in self.indexes}
        return {k: indexes[k].validate_label(v) for k, v in label.items()}

    def reduce(self, documents, labels):
        if not documents:
            return documents

        keys = set(index.name for index in self.indexes)
        keys = keys.intersection(documents[0])
        documents = [hashable_doc(doc) for doc in documents]
        for index in self.indexes:
            if index.name not in labels:
                continue
            others = [k for k in keys if k not in index.names]
            if not others:
                continue
            reduced_documents = []
            for _, docs in toolz.groupby(others, documents).items():
                label = labels[index.name]
                reduced = index.reduce(docs, label)
                reduced_documents.extend(reduced)
            documents = reduced_documents
        documents = [unhashable_doc(doc) for doc in documents]
        return documents

    def __repr__(self):
        return f"MultiIndex({self.indexes})"

