from typing import Dict, Any

import logging
import time
import json
from numpy import ndarray

from pydantic import BaseConfig
from fastapi import FastAPI, APIRouter
import haystack
from haystack import Pipeline
from haystack.schema import Span
from haystack.telemetry import send_event_if_public_demo

from rest_api.utils import get_app, get_pipelines
from rest_api.config import LOG_LEVEL
from rest_api.config import INDEX_NAME
from rest_api.config import CHECKPOINT_PATH
from rest_api.config import QUERY_PIPELINE_NAME
from rest_api.config import QUERY_PIPELINE_NAME
from rest_api.config import BENCHMARK_LOG_TAG
from rest_api.schema import QueryRequest, QueryResponse

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


BaseConfig.arbitrary_types_allowed = True


router = APIRouter()
app: FastAPI = get_app()
query_pipeline: Pipeline = get_pipelines().get("query_pipeline", None)
concurrency_limiter = get_pipelines().get("concurrency_limiter", None)
request_id = 0

@router.get("/initialized")
def check_status():
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@router.get("/hs_version")
def haystack_version():
    """
    Get the running Haystack version.
    """
    return {"hs_version": haystack.__version__}


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    logger.info(f"pipelinename = {QUERY_PIPELINE_NAME}")
    if 'request_id' not in request.params:
        global request_id
        request.params['request_id'] = {'id': str(request_id)}
        request_id = request_id + 1

    with concurrency_limiter.run():
        start_time = time.time()
        if QUERY_PIPELINE_NAME != 'esds_bm25r_colbert' : 
            result = _process_request(query_pipeline, request)
        else : 
            result = _process_request_bm25_colbert(query_pipeline, request)
#        p = QAPipe(request)
#        result = p.get_pipe().process_request(request)
        end_time = time.time()
        logger.info(f"{BENCHMARK_LOG_TAG} {{request_id: {request.params['request_id']['id']}}} {{end2end_time: {(end_time-start_time)}}}")
        return result

def _process_request_bm25_colbert(pipeline, request) -> QueryResponse:
    retriever_topk = request.params['Retriever']['top_k']
    ranker_topk = request.params['Ranker']['top_k']

    logger.info(f"retriever topk={retriever_topk}")
    logger.info(f"ranker topk={ranker_topk}")
    with open('out.log', 'a') as f:
        f.write('received request = %s.\n' % (request))

    # bm25+colbert specific params
    #params_hybrid = {"Retriever": {"top_k": retriever_topk, "index": INDEX_NAME}, "Ranker": {"top_k": ranker_topk}}
    logger.info(f"params={request.params}")
    result = pipeline.run(query=request.query, params=request.params, debug=request.debug)

    for ans in result['answers'][:ranker_topk]:
        # this is to walk-around the Float32 cannot be serialized issue...
        ans.score = float(ans.score)
        # this is to walk-around offsets_in_document as None
        ans.offsets_in_document = ans.offsets_in_document or ans.offsets_in_context
        # return context if answer is null
        ans.answer = ans.answer or ans.context
        # set the right span
        ans.offsets_in_document = [Span(0, len(ans.answer))]
        ans.offsets_in_context = [Span(0, len(ans.context))]

    # convert embedding from ndarray to list according to schema
    for doc in result['documents'][:ranker_topk]:
        doc.embedding = list(doc.embedding) if doc.embedding else None

    # only return topk results
    result['answers'] = result['answers'][:ranker_topk]
    result['documents'] = result['documents'][:ranker_topk]

    #logger.info(
    #    json.dumps({"request": request, "response": result, "time": f"{(end_time - start_time):.2f}"}, default=str))

    return result



@send_event_if_public_demo
def _process_request(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()

    params = request.params or {}
    # FAQ pipeline doesn't need reader
    if request.pipeline == 'faq':
        params.pop('Reader', None)

    # format global, top-level filters (e.g. "params": {"filters": {"name": ["some"]}})
    if "filters" in params.keys():
        params["filters"] = _format_filters(params["filters"])

    # format targeted node filters (e.g. "params": {"Retriever": {"filters": {"value"}}})
    for key in params.keys():
        if "filters" in params[key].keys():
            params[key]["filters"] = _format_filters(params[key]["filters"])

    result = pipeline.run(query=request.query, params=params, debug=request.debug)
    # this is to walk-around the Float32 cannot be serialized issue...
    for ans in result['answers']:
        ans.score = float(ans.score)
        ans.offsets_in_document = ans.offsets_in_document or ans.offsets_in_context

    # convert embedding from ndarray to list according to schema
    for doc in result['documents']:
        doc.embedding = list(doc.embedding) if doc.embedding is not None else None
 
    # Ensure answers and documents exist, even if they're empty lists
    if not "documents" in result:
        result["documents"] = []
    if not "answers" in result:
        result["answers"] = []

    # if any of the documents contains an embedding as an ndarray the latter needs to be converted to list of float
    for document in result["documents"]:
        if isinstance(document.embedding, ndarray):
            document.embedding = document.embedding.tolist()

    #logger.info(
    #    json.dumps({"request": request, "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str)
    #)
    return result


def _format_filters(filters):
    """
    Adjust filters to compliant format:
    Put filter values into a list and remove filters with null value.
    """
    new_filters = {}
    if filters is None:
        logger.warning(
            f"Request with deprecated filter format ('\"filters\": null'). "
            f"Remove empty filters from params to be compliant with future versions"
        )
    else:
        for key, values in filters.items():
            if values is None:
                logger.warning(
                    f"Request with deprecated filter format ('{key}: null'). "
                    f"Remove null values from filters to be compliant with future versions"
                )
                continue

            if not isinstance(values, list):
                logger.warning(
                    f"Request with deprecated filter format ('{key}': {values}). "
                    f"Change to '{key}':[{values}]' to be compliant with future versions"
                )
                values = [values]

            new_filters[key] = values
    return new_filters

'''
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class QAPipeExtractive(metaclass=Singleton):
    def __init__(self):
        # at this moment PIPELINE is needed by other module, so it's globally created...to-be-fixed
        # self.p = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME)
        self.p = PIPELINE
        logger.info(f"Loaded singleton pipeline nodes: {self.p.graph.nodes.keys()}")

    def process_request(self, request) -> QueryResponse:
        return _process_request(self.p, request)


class QAPipeFaq(metaclass=Singleton):
    def __init__(self, mode=0):
        from haystack.nodes.retriever.dense import EmbeddingRetriever, DensePassageRetriever
        from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
        from haystack.document_stores import FAISSDocumentStore
        from haystack.pipelines import FAQPipeline

        self.mode = mode
        self.faiss_ds_idx = 'faiss'

        if self.mode == 0:  # es+embedding_retriever
            logger.info('es+emb FAQPipeline is selected.')
            # assuming document store already has the faq documents indexed.
            document_store = ElasticsearchDocumentStore(host="elasticsearch",
                                                        embedding_field="question_emb",
                                                        embedding_dim=768,
                                                        excluded_meta_data=["question_emb"])
            document_store.debug = True
            retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert")
            retriever.debug = True
            self.p = FAQPipeline(retriever=retriever).pipeline
        elif self.mode == 1:  # faiss+dpr
            logger.info('faiss+dpr FAQPipeline is selected.')
            """ below is only for creating new faiss docstore
            document_store = FAISSDocumentStore(sql_url='sqlite:///faiss-so.db',
                                                faiss_index_factory_str="HNSW",
                                                return_embedding=True,
                                                index=self.faiss_ds_idx)
            """
            document_store = FAISSDocumentStore.load('faiss-index-so.faiss')

            retriever = DensePassageRetriever(document_store=document_store,
                                              query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                              passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                              max_seq_len_query=64,
                                              max_seq_len_passage=256,
                                              batch_size=16,
                                              embed_title=True,
                                              use_fast_tokenizers=True)
            self.p = FAQPipeline(retriever=retriever).pipeline
        elif self.mode == 2:  # milvus+dpr
            pass
        elif self.mode == 3: # bm25+colbert
            logger.info('bm25+colbert FAQPipeline is selected.')
            self.p = self.colbert_pipeline(self.init_doc_store())

        logger.info(f"Loaded singleton pipeline nodes: {self.p.graph.nodes.keys()}")

    def process_request(self, request):
        return _process_request(self.p, request) if self.mode != 3 else _process_request_bm25_colbert(self.p, request)

    def init_doc_store(self):
        from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
        document_store = ElasticsearchDocumentStore(
            host="elasticsearch", username="", password="")
            #host="elasticsearch", username="", password="", index="stackoverflow_paraphrase")
        return document_store

    def init_bm25_retriever(self, document_store, top_k=1000):
        from haystack.nodes.retriever.sparse import ElasticsearchRetriever
        retriever = ElasticsearchRetriever(document_store=document_store, top_k=top_k)
        return retriever

    def colbert_pipeline(self, document_store, top_k=1000):
        from haystack.nodes.ranker.colbert_modeling import ColBERTRanker
        from haystack.nodes.other import Docs2Answers
        model_fullpath = CHECKPOINT_PATH

        retriever = self.init_bm25_retriever(document_store, top_k=top_k)
        reranker = ColBERTRanker(
            model_path=model_fullpath,
            top_k=1000,
            amp=True,
            use_gpu=True,
            batch_size=1024,
        )
        pipe = Pipeline()
        pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
        pipe.add_node(component=reranker, name="Ranker", inputs=["Retriever"])
        pipe.add_node(component=Docs2Answers(), name="Docs2Answers", inputs=["Ranker"])
        return pipe


class QAPipe:
    def __init__(self, request):
        if request.pipeline == 'extractive':
            self.p = QAPipeExtractive()
        elif request.pipeline == 'faq':
            self.p = QAPipeFaq(request.mode)

    def get_pipe(self):
        return self.p
'''

