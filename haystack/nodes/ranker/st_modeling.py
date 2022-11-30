from pathlib import Path
from typing import List, Optional, Union
from haystack.nodes.ranker.base import BaseRanker
from haystack.schema import Document
from sentence_transformers import SentenceTransformer, util


class STRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        top_k: int = 10,
    ):
        self.model = SentenceTransformer(model_name_or_path)
        self.top_k = top_k
        self.model.eval()

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k

        docs = [d.content for d in documents]
        document_embeddings = self.model.encode(docs)
        query_embedding = self.model.encode(query)
        scores = util.cos_sim(query_embedding, document_embeddings).flatten()

        # rank documents according to scores
        sorted_scores_and_documents = sorted(
            zip(scores, documents),
            key=lambda similarity_document_tuple: similarity_document_tuple[0],
            reverse=True,
        )
        sorted_documents = [doc for _, doc in sorted_scores_and_documents]
        return sorted_documents[:top_k]

    def predict_batch(
        self,
        query_doc_list: List[dict],
        top_k: Optional[int] = None,
    ):
        raise NotImplementedError
