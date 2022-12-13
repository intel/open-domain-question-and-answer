from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore

import ray
import base64
import json
import modin.pandas as modin_pd
import pandas as pd
import numpy as np
import argparse, time, os
from modin.config import Engine

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH","/home/user/data/pytorch_model.bin")
FILENAME = {'dev': 'marco/dev_v2.1.json', 'train': 'marco/train_v2.1.json'}

class EmbeddingTask:
    def __init__(self):
        time1 = time.time()
        if config.ds == 'emr':
            from haystack.nodes.retriever.dense import EmbeddingRetriever
            time2 = time.time()
            document_store = ElasticsearchDocumentStore(host=config.host, username="", password="",
                                                    index="document",
                                                    embedding_field="question_emb",
                                                    embedding_dim=768,
                                                    excluded_meta_data=["question_emb"])

            self.retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert",
                                   use_gpu=True)
        else:
            from haystack.nodes.ranker.colbert_modeling import ColBERTRanker
            time2 = time.time()
            self.ranker = ColBERTRanker(model_path=CHECKPOINT_PATH, use_gpu=True, batch_size=1)
        
        print(f'import python package time={time2-time1}')

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        # get embeddings for question
        questions = list(batch["Query"].values)
        time1 = time.time()
        if config.ds == 'emr':
            batch["question_emb"] = self.retriever.embed_queries(texts=questions)
            time2 = time.time()
            print(f'emr embedding time={time2-time1}')
        else:
            emb_array = self.ranker._encode_multiple_docs(docs=questions, batch_size=1).cpu().numpy()
            time2 = time.time()
            batch["colbert_emb"] = self.encode(emb_array)
            time3 = time.time()
            print(f'colbert embedding time={time2-time1}')
        return batch

    def encode(self, arr: np.ndarray) -> str:
        dt = np.dtype(arr.dtype)
        dt = dt.newbyteorder(">")
        arr_be = arr.astype(dt)
        return base64.b64encode(arr_be.tobytes()).decode()


class IndexingTask:
    def __init__(self):
        if config.ds == 'emr':
            marco_mapping = {"mappings": {
                "properties": {
                    "content":          { "type": "text"  },
                    "Query_id":         { "type": "integer" },
                    "Query":            { "type": "text", "index": "false" },
                    "answer":           { "type": "text", "index": "false"  },
                    "question_emb":     {"type": "dense_vector", "dims": 768 },
                } }
            }
            self.document_store = ElasticsearchDocumentStore(host=config.host, username="", password="",
                                                    index="document",
                                                    custom_mapping=marco_mapping,
                                                    embedding_field="question_emb",
                                                    embedding_dim=768,
                                                    excluded_meta_data=["question_emb"])

        else:
            marco_mapping = {"mappings": {
                "properties": {
                    "content":          { "type": "text"  },
                    "Query_id":         { "type": "integer" },
                    "Query":            { "type": "text", "index": "false" },
                    "answer":           { "type": "text", "index": "false"  },
                    "colbert_emb":      { "type": "binary", "index": "false" },
                } }
            }
            self.document_store = ElasticsearchDocumentStore(host=config.host, username="", password="",
                                                    index="marco_colbert",
                                                    custom_mapping=marco_mapping)


    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        # write into document store
        print('write into documentstore...')
        docs_to_index = batch.to_dict(orient="records")
        rest_keys = list(docs_to_index[0].keys())
        docs = [{'content': d['Query'], 'meta': {k: d[k] for k in rest_keys}} for d in docs_to_index]
        time1 = time.time()
        self.document_store.write_documents(docs)
        time2 = time.time()
        consume = time2 - time1
        print(f'colbert or emr write docs time={consume}')
        return batch


class GenDocStoreBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.qa = None

    def __call__(self, *args, **kwargs):
        ts_start = time.time()
        self.prepare_data()
        ts_pre_data = time.time()
        print('prepare_data cost: %.4f sec.' % (ts_pre_data - ts_start))
        self.gen_docstore_es()
        print('gen_docstore cost: %.4f sec.' % (time.time() - ts_pre_data))

    def prepare_data(self):
        print("prepare data start")
        query_list = []
        query_id_list = []
        wellFormedAnswers_list = []
        answers_list = []
        passages_list = []

        with open(FILENAME[self.cfg.mode]) as json_file:
            dataList = json_file.readlines()
            for data in dataList:
                json_data = json.loads(data)
            print(json_data.keys())

        for query in json_data['query'].values():
            query_list.append(query)
        for query_id in json_data['query_id'].values():
            query_id_list.append(query_id)
        for wellFormedAnswer in json_data['wellFormedAnswers'].values():
            wellFormedAnswers_list.append(str(wellFormedAnswer))
        for answer in json_data['answers'].values():
            answers_list.append(str(answer))
        for passage in json_data['passages'].values():
            passages_list.append(str(passage))

        data = {}
        for i in range(len(query_list)):
            if wellFormedAnswers_list[i] == '[]':
                if answers_list[i] == "['No Answer Present.']":
                    data[query_list[i]] = passages_list[i]
                else:
                    data[query_list[i]] = answers_list[i]
            else:
                data[query_list[i]] = wellFormedAnswers_list[i]

        self.qa = modin_pd.DataFrame(list(data.items()), columns=['Query', 'answer'])
        self.qa['Query_id'] = query_id_list
        print(self.qa.columns)

    def gen_docstore_es(self):
        print('launching es & retriever...')
        ds = ray.data.from_modin(self.qa)
        time1 = time.time()
        ds = ds.map_batches(EmbeddingTask, batch_format="pandas", batch_size=1, compute="actors")
        ds.show(1)
        time2 = time.time()
        print(f"embedding total time ={time2 - time1}")
        print(ds.count())
        ds.map_batches(IndexingTask, batch_format="pandas", batch_size=50, compute="actors")
        time3 = time.time()
        print(f"write doc store total time ={time3 - time2}")


def parse_cmd():
    desc = 'generate documentstore for marco dataset...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-m', type=str, default='train', dest='mode', help='dev or train', choices=['dev', 'train'])
    args.add_argument('-d', type=str, default='emr', dest='ds', help='type of pipeline', choices=['emr', 'colbert'])
    args.add_argument('-p', type=str, default='localhost', dest='host', help='Ip address of elasticsearch host')
    return args.parse_args()


if __name__ == "__main__":
    Engine.put("ray")
    print("Connect ray head!!!!")
    ray.init(address='auto')
    config = parse_cmd()
    GenDocStoreBase(config)()

