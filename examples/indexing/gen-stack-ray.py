
import ray
import os
import base64
import numpy as np
from ray.data import ActorPoolStrategy
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
import kaggle

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH","/home/user/data/colbert.dnn")
import modin.pandas as modin_pd
import pandas as pd
from bs4 import BeautifulSoup
import argparse, time, os
import time
from modin.config import Engine

SO_PATH = './stack-overflow/'
SO_FILE = [{'question': 'Questions.csv', 'answer': 'Answers.csv'},
           {'question': 'Questions-S.csv', 'answer': 'Answers-S.csv'}]


#@ray.remote
def process_emr_data(df: pd.DataFrame)->pd.DataFrame:
    print('start to process answers...')
    print(df.shape)
    print('import python package...')
    time1 = time.time()
    from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
    from haystack.nodes.retriever.dense import EmbeddingRetriever
    time2 = time.time()
    print(f'import python package time={time2-time1}')
    '''stackoverflow_mapping == {
                "mappings": {
                    "properties": {self.name_field: {"type": "keyword"}, self.content_field: {"type": "text"}},
                    "dynamic_templates": [
                        {"strings": {"path_match": "*", "match_mapping_type": "string", "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {"analysis": {"analyzer": {"default": {"type": self.analyzer}}}},
            }'''
    document_store = ElasticsearchDocumentStore(host=config.host, username="", password="",
                                                index="stackoverflow",
                                                embedding_field="question_emb",
                                                embedding_dim=768,
                                                excluded_meta_data=["question_emb"])

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert",
                                    use_gpu=True)

    # get embeddings for question
    questions = list(df["text"].values)
    print('total embeddings len = %d...' % len(questions))
    df["question_emb"] = retriever.embed_queries(texts=questions)

    # write into document store
    print('write into documentstore...')
    docs_to_index = df.to_dict(orient="records")
    print(df.columns)
    print(docs_to_index)
    rest_keys = list(docs_to_index[0].keys())
    rest_keys.remove('text')
    docs = [{'content': d['text'], 'meta': {k: d[k] for k in rest_keys}} for d in docs_to_index]
    document_store.write_documents(docs)
    return len(questions)

class EmbeddingTask :
    def __init__(self):
        print('import python package...')
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
        questions = list(batch["text"].values)
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

class IndexingTask :
    def __init__(self):
        if config.ds == 'emr':
            stackoverflow_mapping = {"mappings": {
                "properties": {
                    "content":          { "type": "text"  },
                    "Question_id":      { "type": "integer" },
                    "question-body":    { "type": "keyword", "index": "false" },
                    "answer":           { "type": "keyword", "index": "false"  },
                    "question_emb":     {"type": "dense_vector", "dims": 768 },
                } }
            }
            self.document_store = ElasticsearchDocumentStore(host=config.host, username="", password="",
                                                    index="document",
                                                    custom_mapping=stackoverflow_mapping,
                                                    embedding_field="question_emb",
                                                    embedding_dim=768,
                                                    excluded_meta_data=["question_emb"])

        else:
            stackoverflow_mapping = {"mappings": {
                "properties": {
                    "content":          { "type": "text"  },  
                    "Question_id":      { "type": "integer" }, 
                    "question-body":    { "type": "text", "index": "false" },    
                    "answer":           { "type": "text", "index": "false"  },    
                    "colbert_emb":      { "type": "binary", "index": "false" },     
                } }
            }
            self.document_store = ElasticsearchDocumentStore(host=config.host, username="", password="",
                                                    index="stackoverflow_colbert",
                                                    custom_mapping=stackoverflow_mapping)



    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        # write into document store
        docs_to_index = batch.to_dict(orient="records")
        rest_keys = list(docs_to_index[0].keys())
        rest_keys.remove('text')
        try:
            docs = [{'content': d['text'], 'meta': {k: d[k] for k in rest_keys}} for d in docs_to_index]
            time1 = time.time()
            self.document_store.write_documents(docs)
            time2 = time.time()
            consume = time2 - time1
            print(f'colbert or emr write docs time={consume}')
        except Exception as e:
            print(f"Exception Info:{e}")
        return batch



class GenDocStoreBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.qa = None
        self.faiss_ds_idx = 'faiss'

    def __call__(self, *args, **kwargs):

        if self.cfg.colbert == 'ColBERT':
            self.test_colbert()
            return

        ts_start = time.time()
        self.prepare_data()
        ts_pre_data = time.time()
        print('prepare_data cost: %.4f sec.' % (ts_pre_data - ts_start))
        self.gen_docstore_es()
        print('gen_docstore cost: %.4f sec.' % (time.time() - ts_pre_data))

    def prepare_data(self):
        if self.cfg.mode == 0 and self.cfg.force_download:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('stackoverflow/stacksample', path=SO_PATH, unzip=True, quiet=False)

        print('start to read answers...')
        start = time.time()
        answers = modin_pd.read_csv(SO_PATH + SO_FILE[self.cfg.mode]['answer'], encoding="ISO-8859-1")
        time1 = time.time()
        print(f"read answer csv time ={time1-start}");
        answers.drop(columns=['OwnerUserId', 'CreationDate', 'Id'], inplace=True)
        time2 = time.time()
        print(f"answer drop time ={time2-time1}");
        answers = ray.data.from_modin(answers)
        time3 = time.time()
        print(f"convert pandas to ray df time ={time3-time2}");
        #answers = answers.repartition(80)
        print('start to process answers...')
        top_answers = answers.groupby("ParentId").map_groups(lambda g: g[g['Score'] == g['Score'].max()])
        #top_answers = answers.groupby("ParentId").apply(lambda g: g[g['Score'] == g['Score'].max()])
        time4 = time.time()
        print(f"answer groupby time ={time4-time3}");
        top_answers = top_answers.to_modin()
        time5 = time.time()
        print(f"convert ray df to modin time ={time5-time4}");
        top_answers.rename(columns={"ParentId": "Id", "Body": "TopAnswer-html"}, inplace=True)
        time6 = time.time()
        print(f"answer rename time ={time6-time5}");
        top_answers.drop(columns=['Score'], inplace=True)
        time7 = time.time()
        print(f"answer drop time ={time7-time6}");

        print('start to process questions...')
        questions = modin_pd.read_csv(SO_PATH + SO_FILE[self.cfg.mode]['question'], encoding="ISO-8859-1")
        time8 = time.time()
        print(f"read questions csv time ={time8-time7}");
        questions.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate', 'Score'], inplace=True)
        time9 = time.time()
        print(f"questions drop time ={time9-time8}");
        questions.rename(columns={"Title": "text", "Body": "Question-html"}, inplace=True)
        time10 = time.time()
        print(f"questions rename time ={time10-time9}");
        self.qa = questions.merge(top_answers, on='Id')
        time11 = time.time()
        print(f"questions answer merge time ={time11-time10}");
        #self.qa.drop(columns=['Id'], inplace=True)
        self.qa.rename(columns={'Id':'Question_id'}, inplace=True)
        time12 = time.time()
        print(f"qa drop time ={time12-time11}");

        # now cleanup if any
        print('Dupplicate entries: {}'.format(self.qa.duplicated().sum()))
        self.qa.drop_duplicates(inplace=True)
        time13 = time.time()
        print(f"qa drop duplicates time ={time13-time12}");
 
        # convert html to plain text
        print('convert html to plain text...')
        self.qa['question-body'] = self.qa['Question-html'].apply(lambda x: BeautifulSoup(x).get_text())
        self.qa['answer'] = self.qa['TopAnswer-html'].apply(lambda x: BeautifulSoup(x).get_text())
        self.qa.drop(columns=['Question-html', 'TopAnswer-html'], inplace=True)
        time14 = time.time()
        print(f"convert html time ={time14-time13}");
        print(self.qa.columns)

          
   

    def gen_docstore_es(self):
        print('launching es & retriever...')
        ds = ray.data.from_modin(self.qa)
        time1 = time.time()
        batch_size = 1
        if self.cfg.ds == 'emr':
            batch_size = self.cfg.bs
        ds = ds.map_batches(EmbeddingTask, batch_format="pandas", batch_size=batch_size, compute="actors")
        ds.show(1)
        time2 = time.time()
        print(f"embedding time ={time2 - time1}")
        print(ds.count())
        ds.map_batches(IndexingTask, batch_format="pandas", batch_size=50, compute="actors")
        time3 = time.time()
        print(f"write doc store total time ={time3 - time2}")
       
        # give a simple test
        '''from haystack.pipelines import FAQPipeline
        pipe = FAQPipeline(retriever=retriever)

        prediction = pipe.run(query="Python - how to get current date?", params={"Retriever": {"top_k": 3}})
        print_answers(prediction, details="all")'''



def parse_cmd():
    desc = 'generate documentstore for stack-overflow dataset...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-m', type=int, default=0, dest='mode', help='0: actual, 1: debug', choices=[0, 1])
    args.add_argument('-f', dest='force_download', help='force download so-ds', action='store_true', default=False)
    args.add_argument('-d', type=str, default='emr', dest='ds', help='type of documentstore', choices=['emr'])
    args.add_argument('-b', type=int, default=16, dest='bs', help='batch size for EMR')
    args.add_argument('-ip', type=str, default='localhost', dest='host', help='Ip address of elasticsearch host')
    return args.parse_args()


if __name__ == "__main__":
    Engine.put("ray")  # Modin will use Ray
    print("Connect ray head!!!!")
    ray.init(address='auto')
    config = parse_cmd()
    gen_doc = GenDocStoreBase(config)()
