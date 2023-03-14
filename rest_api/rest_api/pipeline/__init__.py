from typing import Any, Dict

import os
import logging
from pathlib import Path

from haystack.pipelines.base import Pipeline
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.errors import PipelineConfigError

from rest_api.controller.utils import RequestLimiter
from haystack.nodes.retriever.dense import EmbeddingRetriever, DensePassageRetriever, ColBERTRetriever
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores.plaid import PLAIDDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.nodes.ranker.colbert_modeling import ColBERTRanker
from haystack.nodes.other import Docs2Answers
from haystack.nodes.retriever.sparse import ElasticsearchRetriever



logger = logging.getLogger(__name__)

# Since each instance of FAISSDocumentStore creates an in-memory FAISS index, the Indexing & Query Pipelines would
# end up with different indices. The same applies for InMemoryDocumentStore.
UNSUPPORTED_DOC_STORES = (FAISSDocumentStore, InMemoryDocumentStore)


def setup_pipelines() -> Dict[str, Any]:
    # Re-import the configuration variables
    from rest_api import config  # pylint: disable=reimported
    pipelines = {}
    if config.QUERY_PIPELINE_NAME == 'query' :
        # Load query pipeline
        query_pipeline = Pipeline.load_from_yaml(Path(config.PIPELINE_YAML_PATH), pipeline_name=config.QUERY_PIPELINE_NAME)
        logging.info(f"Loaded pipeline nodes: {query_pipeline.graph.nodes.keys()}")
        pipelines["query_pipeline"] = query_pipeline

        # Find document store
        document_store = query_pipeline.get_document_store()
        logging.info(f"Loaded docstore: {document_store}")
        pipelines["document_store"] = document_store
        # Load indexing pipeline (if available)
        try:
            indexing_pipeline = Pipeline.load_from_yaml(
                Path(config.PIPELINE_YAML_PATH), pipeline_name=config.INDEXING_PIPELINE_NAME
            )
            docstore = indexing_pipeline.get_document_store()
            if isinstance(docstore, UNSUPPORTED_DOC_STORES):
                indexing_pipeline = None
                raise PipelineConfigError(
                    "Indexing pipelines with FAISSDocumentStore or InMemoryDocumentStore are not supported by the REST APIs."
                )

        except PipelineConfigError as e:
            indexing_pipeline = None
            logger.error(f"{e.message}\nFile Upload API will not be available.")

        finally:
            pipelines["indexing_pipeline"] = indexing_pipeline

    elif config.QUERY_PIPELINE_NAME == 'esds_bm25r_colbert' :
        logger.info('colbert FAQPipeline is selected.')
        model_fullpath = config.CHECKPOINT_PATH
        logger.info(f'colbert  model = {model_fullpath}.')

        document_store = ElasticsearchDocumentStore(host=config.DOCUMENTSTORE_PARAMS_HOST,
                                                    port= config.DOCUMENTSTORE_PARAMS_PORT,
                                                    username="", password="",index=config.INDEX_NAME)

        retriever = ElasticsearchRetriever(document_store=document_store, top_k=1000)
        reranker = ColBERTRanker(
            model_path=model_fullpath,
            top_k=1000,
            amp=True,
            batch_size=1024,
        )
        query_pipeline = Pipeline()
        query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        query_pipeline.add_node(component=reranker, name="Ranker", inputs=["Retriever"])
        query_pipeline.add_node(component=Docs2Answers(), name="Docs2Answers", inputs=["Ranker"])
        pipelines["query_pipeline"] = query_pipeline
        pipelines["document_store"] = document_store

    # Setup concurrency limiter
    concurrency_limiter = RequestLimiter(config.CONCURRENT_REQUEST_PER_WORKER)
    logging.info("Concurrent requests per worker: {CONCURRENT_REQUEST_PER_WORKER}")
    pipelines["concurrency_limiter"] = concurrency_limiter

   
    # Create directory for uploaded files
    os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)
    # non-YAML pipleline


    return pipelines
