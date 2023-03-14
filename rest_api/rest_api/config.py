import os
from pathlib import Path


PIPELINE_YAML_PATH = os.getenv(
    "PIPELINE_YAML_PATH", str((Path(__file__).parent / "pipeline" / "pipelines.haystack-pipeline.yml").absolute())
)
QUERY_PIPELINE_NAME = os.getenv("QUERY_PIPELINE_NAME", "query")
INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")
INDEX_NAME = os.getenv("INDEX_NAME", "document")
DOCUMENTSTORE_PARAMS_HOST = os.getenv("DOCUMENTSTORE_PARAMS_HOST", "elasticsearch")
DOCUMENTSTORE_PARAMS_PORT = os.getenv("DOCUMENTSTORE_PARAMS_PORT", "9200")
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent / "file-upload").absolute()))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH","/home/user/data/colbert.dnn")
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH","/home/user/data/faiss-index-so.faiss")
MODEL_PATH = os.getenv("MODEL_PATH", "/home/user/model")
PLAID_INDEX_PATH = os.getenv("PLAID_INDEX_PATH", "/home/user/data/plaid_indexing/")
PLAID_COLLECTION_PATH=os.getenv("PLAID_COLLECTION_PATH", "/home/user/data/psgs_w100.tsv")
CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", "4"))
