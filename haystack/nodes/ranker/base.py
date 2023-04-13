from typing import List, Optional, Union

import logging
from abc import abstractmethod
from functools import wraps
from time import perf_counter

from haystack.schema import Document
from haystack.nodes.base import BaseComponent
from haystack.errors import HaystackError, PipelineError


logger = logging.getLogger(__name__)


class BaseRanker(BaseComponent):
    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0
    index_count = 0
    index_time = 0.0

    @abstractmethod
    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        pass

    @abstractmethod
    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Union[List[Document], List[List[Document]]]:
        pass

    def run(
        self,
        root_node: str,
        query: Optional[str] = None,
        documents: Optional[List[Document]] = None,
        top_k: Optional[int] = None):  # type: ignore

        if root_node == "Query":
            if query is None:
                raise HaystackError(
                    "Must provide a 'query' parameter for rankers in pipelines where Query is the root node."
                )
            if not isinstance(query, str):
                logger.error(
                    "The ranker received an unusual query: '%s' This query is likely to produce garbage output.",
                    query,
                )

            self.query_count += 1
            if documents:
                predict = self.timing(self.predict, "query_time")
                results = predict(query=query, documents=documents, top_k=top_k)
            else:
                results = []

            document_ids = [doc.id for doc in results]
            logger.debug("Retrieved documents with IDs: %s", document_ids)
            output = {"documents": results}
            stream = "output_1" 

        elif root_node == "File":
            self.index_count += len(documents) if documents else 0
            run_indexing = self.timing(self.run_indexing, "index_time")
            output, stream = run_indexing(documents=documents)

        else:
            raise PipelineError(f"Invalid root_node '{root_node}'.")

        return output, stream

    def run_batch(  # type: ignore
        self,
        root_node: str,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        if root_node == "Query":
            if queries is None:
                raise HaystackError(
                    "Must provide a 'queries' parameter for retrievers in pipelines where Query is the root node."
                )
            if not all(isinstance(query, str) for query in queries):
                logger.error(
                    "The retriever received an unusual list of queries: '%s' Some of these queries are likely to produce garbage output.",
                    queries,
                )
            self.query_count = +len(queries)
            predict_batch = self.timing(self.predict_batch, "query_time")
            results = predict_batch(queries=queries, documents=documents, top_k=top_k, batch_size=batch_size)

            for doc_list in results:
                document_ids = [doc.id for doc in doc_list]
                logger.debug("Ranked documents with IDs: %s", document_ids)

            output = {"documents": results}
            stream = "output_1"

        elif root_node == "File":
            self.index_count += len(documents) if documents else 0
            run_indexing = self.timing(self.run_indexing, "index_time")
            output, stream = run_indexing(documents=documents)

        else:
            raise PipelineError(f"Invalid root_node '{root_node}'.")
        return output, stream

    def run_indexing(self, documents: List[Document]):
        output = {"documents": documents}
        return output, "output_1"

    def timing(self, fn, attr_name):
        """Wrapper method used to time functions."""

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if attr_name not in self.__dict__:
                self.__dict__[attr_name] = 0
            tic = perf_counter()
            ret = fn(*args, **kwargs)
            toc = perf_counter()
            self.__dict__[attr_name] += toc - tic
            return ret

        return wrapper

    def print_time(self):
        print("Ranker (Speed)")
        print("---------------")
        if not self.index_count:
            print("No indexing performed via Ranker.run()")
        else:
            print(f"Documents indexed: {self.index_count}")
            print(f"Index time: {self.index_time}s")
            print(f"{self.query_time / self.query_count} seconds per document")
        if not self.query_count:
            print("No querying performed via Retriever.run()")
        else:
            print(f"Queries Performed: {self.query_count}")
            print(f"Query time: {self.query_time}s")
            print(f"{self.query_time / self.query_count} seconds per query")

    def eval(
        self,
        label_index: str = "label",
        doc_index: str = "eval_document",
        label_origin: str = "gold_label",
        top_k: int = 10,
        open_domain: bool = False,
        return_preds: bool = False,
    ) -> dict:
        """
        Performs evaluation of the Ranker.
        Ranker is evaluated in the same way as a Retriever based on whether it finds the correct document given the query string and at which
        position in the ranking of documents the correct document is.

        Returns a dict containing the following metrics:

            - "recall": Proportion of questions for which correct document is among retrieved documents
            - "mrr": Mean of reciprocal rank. Rewards retrievers that give relevant documents a higher rank.
              Only considers the highest ranked relevant document.
            - "map": Mean of average precision for each question. Rewards retrievers that give relevant
              documents a higher rank. Considers all retrieved relevant documents. If ``open_domain=True``,
              average precision is normalized by the number of retrieved relevant documents per query.
              If ``open_domain=False``, average precision is normalized by the number of all relevant documents
              per query.

        :param label_index: Index/Table in DocumentStore where labeled questions are stored
        :param doc_index: Index/Table in DocumentStore where documents that are used for evaluation are stored
        :param top_k: How many documents to return per query
        :param open_domain: If ``True``, retrieval will be evaluated by checking if the answer string to a question is
                            contained in the retrieved docs (common approach in open-domain QA).
                            If ``False``, retrieval uses a stricter evaluation that checks if the retrieved document ids
                            are within ids explicitly stated in the labels.
        :param return_preds: Whether to add predictions in the returned dictionary. If True, the returned dictionary
                             contains the keys "predictions" and "metrics".
        """
        raise NotImplementedError
