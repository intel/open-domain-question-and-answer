
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.retriever.dense import EmbeddingRetriever, DensePassageRetriever, ColBERTRetriever
from haystack.utils import print_answers
from haystack.nodes.reader.farm import FARMReader

#import kaggle
import pandas as pd
from bs4 import BeautifulSoup
import argparse, time, os

SO_PATH = './stack-overflow/'
SO_FILE = [{'question': 'Questions.csv', 'answer': 'Answers.csv'},
           {'question': 'Questions-S.csv', 'answer': 'Answers-S.csv'}]

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
        self.gen_docstore_es() if self.cfg.ds == 'es' else self.gen_docstore_faiss()
        print('gen_docstore cost: %.4f sec.' % (time.time() - ts_pre_data))

    def prepare_data(self):
        if self.cfg.mode == 0 and self.cfg.force_download:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('stackoverflow/stacksample', path=SO_PATH, unzip=True, quiet=False)

        print(f"prepare data start")
        start = time.time()
        answers = pd.read_csv(SO_PATH + SO_FILE[self.cfg.mode]['answer'], encoding="ISO-8859-1")
        time1=time.time()
        answers.drop(columns=['OwnerUserId', 'CreationDate', 'Id'], inplace=True)
        time2=time.time()

        print('start to process answers...')
        top_answers = answers.groupby("ParentId").apply(lambda g: g[g['Score'] == g['Score'].max()])
        time3=time.time()
        top_answers.reset_index(drop=True, inplace=True)
        time4=time.time()
        top_answers.rename(columns={"ParentId": "Id", "Body": "TopAnswer-html"}, inplace=True)
        time5=time.time()
        top_answers.drop(columns=['Score'], inplace=True)
        time6=time.time()

        print('start to process questions...')
        questions = pd.read_csv(SO_PATH + SO_FILE[self.cfg.mode]['question'], encoding="ISO-8859-1")
        time7=time.time()
        questions.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate', 'Score'], inplace=True)
        time8=time.time()
        questions.rename(columns={"Title": "text", "Body": "Question-html"}, inplace=True)
        time9=time.time()
        self.qa = questions.merge(top_answers, on='Id')
        time10=time.time()
        self.qa.drop(columns=['Id'], inplace=True)
        time11=time.time()

        # now cleanup if any
        print('Duplicate entries: {}'.format(self.qa.duplicated().sum()))
        self.qa.drop_duplicates(inplace=True)
        time12=time.time()

        # convert html to plain text
        print('convert html to plain text...')
        self.qa['question-body'] = self.qa['Question-html'].apply(lambda x: BeautifulSoup(x, features='lxml').get_text())
        self.qa['answer'] = self.qa['TopAnswer-html'].apply(lambda x: BeautifulSoup(x, features='lxml').get_text())
        time13=time.time()
        print(f"prepare data time ={time13-start}")
        print(f"read answer time ={time1-start}")
        print(f"drop time ={time2-time1}")
        print(f"groupby time ={time3-time2}")
        print(f"reset-index time ={time4-time3}")
        print(f"rename time ={time5-time4}")
        print(f"drop time ={time6-time5}")
        print(f"read question time ={time7-time6}")
        print(f"question drop time ={time8-time7}")
        print(f"question rename time ={time9-time8}")
        print(f"question merge time ={time10-time9}")
        print(f"question drop time ={time11-time10}")
        print(f"question drop dup time ={time12-time11}")
        print(f"convert html time ={time13-time12}")


    def gen_docstore_es(self):
        print('launching es & retriever...')
        if self.cfg.start_es:
            from haystack.utils import launch_es
            launch_es()

        document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                                    index="document",
                                                    embedding_field="question_emb",
                                                    embedding_dim=768,
                                                    excluded_meta_data=["question_emb"])

        retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert",
                                       use_gpu=True)

        # get embeddings for question
        print('generating embeddings len = %d ...' % (len(self.qa["text"])))
        questions = list(self.qa["text"].values)
        print('total embeddings len = %d...' % len(questions))
        self.qa["question_emb"] = retriever.embed_queries(texts=questions)

        # write into document store
        print('write into documentstore...')
        docs_to_index = self.qa.to_dict(orient="records")
        rest_keys = list(docs_to_index[0].keys())
        rest_keys.remove('text')
        docs = [{'content': d['text'], 'meta': {k: d[k] for k in rest_keys}} for d in docs_to_index]
        document_store.write_documents(docs)

        # give a simple test
        from haystack.pipelines import FAQPipeline
        pipe = FAQPipeline(retriever=retriever)

        prediction = pipe.run(query="Python - how to get current date?", params={"Retriever": {"top_k": 3}})
        print_answers(prediction, details="all")

    def gen_docstore_faiss(self):
        if os.path.isfile('faiss-index-so.faiss'):
            # to load the document_store, use below class method
            document_store = FAISSDocumentStore.load('faiss-index-so.faiss')
            retriever = DensePassageRetriever(document_store=document_store,
                                              query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                              passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                              max_seq_len_query=64,
                                              max_seq_len_passage=256,
                                              batch_size=self.cfg.bs,
                                              use_gpu=True,
                                              embed_title=True,
                                              use_fast_tokenizers=True)
        else:
            document_store = FAISSDocumentStore(sql_url='postgresql://postgres:postgres@postsql-db:5432/haystack',
                                                faiss_index_factory_str="HNSW",
                                                return_embedding=False,
                                                index=self.faiss_ds_idx)

            # write into document store
            print('write into documentstore...')
            docs_to_index = self.qa.to_dict(orient="records")
            rest_keys = list(docs_to_index[0].keys())
            rest_keys.remove('text')
            docs = [ {'content': d['text'], 'meta': {k: d[k] for k in rest_keys}} for d in docs_to_index]

            document_store.write_documents(docs, index=self.faiss_ds_idx)

            # updating embeddings for question
            print('updating embeddings len = %d ...' % (len(self.qa["text"])))
            retriever = DensePassageRetriever(document_store=document_store,
                                              query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                              passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                              max_seq_len_query=64,
                                              max_seq_len_passage=256,
                                              batch_size=self.cfg.bs,
                                              use_gpu=True,
                                              embed_title=True,
                                              use_fast_tokenizers=True)
            document_store.update_embeddings(retriever, index=self.faiss_ds_idx)
            document_store.save('faiss-index-so.faiss')

        # give a simple test using FARMReader
        from haystack.pipelines import ExtractiveQAPipeline
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
        pipe = ExtractiveQAPipeline(reader, retriever)
        prediction = pipe.run(query="Python - how to get current date?",
                              params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
        print_answers(prediction, details="all")

        # give another try with FAQPipe
        from haystack.pipelines import FAQPipeline
        pipe = FAQPipeline(retriever=retriever)
        prediction = pipe.run(query="Python - how to get current date?", params={"Retriever": {"top_k": 3}})
        print_answers(prediction, details="all")

    def test_colbert(self):
        retriever = ColBERTRetriever(
            root='haystack/nodes/retriever/ColBERT/experiments/',
            experiment='STACK-OVERFLOW',
            checkpoint='haystack/nodes/retriever/ColBERT/experiments/STACK-OVERFLOW/train.py/stack.overflow.fine.tuning/checkpoints/colbert.dnn',
            collection='haystack/nodes/retriever/ColBERT/stackoverflow_paraphrase/collection_zero_idx.tsv',
            index_root='haystack/nodes/retriever/ColBERT/indexes/',
            index_name='STACK-OVERFLOW',
            faiss_depth=1024,
            top_k=10
        )

        from haystack.pipelines import FAQPipeline
        pipe = FAQPipeline(retriever=retriever)
        prediction = pipe.run(query="Python - how to get current date?", params={"Retriever": {"top_k": 3}})
        print_answers(prediction, details="all")


def parse_cmd():
    desc = 'generate documentstore for stack-overflow dataset...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-m', type=int, default=0, dest='mode', help='0: actual, 1: debug', choices=[0, 1])
    args.add_argument('-f', dest='force_download', help='force download so-ds', action='store_true', default=False)
    args.add_argument('-d', type=str, default='faiss', dest='ds', help='type of documentstore', choices=['es', 'faiss'])
    args.add_argument('-b', type=int, default=16, dest='bs', help='batch size for DPR')
    args.add_argument('-c', type=str, default=None, dest='colbert', help='test colbert retriever')
    args.add_argument('-r', type=int, default=1, dest='start_es', help='start elasticsearch docker')
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    GenDocStoreBase(config)()
