from collections.abc import Mapping

from toolz import groupby
from .base import BaseIndex
from ..utils import hashable_doc, unhashable_doc


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

    def reduce(self, docs, labels):
        if not docs:
            return docs

        keys = set(index.name for index in self.indexes)
        keys = keys.intersection(docs[0])
        docs = [hashable_doc(doc) for doc in docs]
        for index in self.indexes:
            if index.name not in labels:
                continue
            others = [k for k in keys if k not in index.names]
            if not others:
                continue
            reduced_documents = []
            for grp in groupby(others, docs).values():
                label = labels[index.name]
                reduced = index.reduce(grp, label)
                reduced_documents.extend(reduced)
            docs = reduced_documents
        docs = [unhashable_doc(doc) for doc in docs]
        return docs

    def __repr__(self):
        return f"MultiIndex({self.indexes})"

    def label_options(self, query):
        label_options = [idx.label_options(query) for idx in self.indexes]
        return label_options
