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
BENCHMARK_LOG_TAG = os.getenv("BENCHMARK_LOG_TAG", "Haystack Benchmark:")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH","/home/user/data/colbert.dnn")
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH","/home/user/data/faiss-index-so.faiss")
CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", "4"))
