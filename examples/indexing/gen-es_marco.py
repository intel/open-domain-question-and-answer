from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes.retriever.dense import EmbeddingRetriever, DensePassageRetriever, ColBERTRetriever
from haystack.utils import print_answers
from haystack.nodes.reader.farm import FARMReader

import json
import pandas as pd
import argparse, time, os

FILENAME = {'dev': 'macro/dev_v2.1.json', 'train': 'macro/train_v2.1.json'}

class GenDocStoreBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.qa = None
        self.faiss_ds_idx = 'faiss'

    def __call__(self, *args, **kwargs):

        ts_start = time.time()
        self.prepare_data()
        ts_pre_data = time.time()
        print('prepare_data cost: %.4f sec.' % (ts_pre_data - ts_start))
        print('--------self.cfg.ds----------', self.cfg.ds, type(self.cfg.ds))
        self.gen_docstore_es() if self.cfg.ds == 'es' else self.gen_docstore_faiss()
        print('gen_docstore cost: %.4f sec.' % (time.time() - ts_pre_data))


    def prepare_data(self):
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
    
        self.qa = pd.DataFrame(list(data.items()), columns=['Query', 'answer'])
        self.qa['Query_id'] = query_id_list
        #return qa_data

    def gen_docstore_es(self):
        print('launching es & retriever...')
        document_store = ElasticsearchDocumentStore(host="elasticsearch", username="", password="",
                                                    index="document",
                                                    embedding_field="question_emb",
                                                    embedding_dim=768,
                                                    excluded_meta_data=["question_emb"])

        retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert",
                                   use_gpu=True)
        
        # get embeddings for question
        print('generating embeddings len = %d ...' % (len(self.qa["Query"])))
        questions = list(self.qa["Query"].values)
        print('total embeddings len = %d...' % len(questions))
        self.qa["question_emb"] = retriever.embed_queries(texts=questions)
    
        # write into document store
        print('write into documentstore...')
        docs_to_index = self.qa.to_dict(orient="records")
        rest_keys = list(docs_to_index[0].keys())
        docs = [{'content': d['Query'], 'meta': {k: d[k] for k in rest_keys}} for d in docs_to_index]
        document_store.write_documents(docs)

        # give a simple test
        from haystack.pipelines import FAQPipeline
        pipe = FAQPipeline(retriever=retriever)
        prediction = pipe.run(query="sum of squares of even numbers formula", params={"Retriever": {"top_k": 3},"request_id": {"id": 0}})
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
            docs = [{'content': d['Query'], 'meta': {k: d[k] for k in rest_keys}} for d in docs_to_index]
            document_store.write_documents(docs, index=self.faiss_ds_idx)

            # updating embeddings for question
            print('generating embeddings len = %d ...' % (len(self.qa["Query"])))
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
        prediction = pipe.run(query="sum of squares of even numbers formula", params={"Retriever": {"top_k": 3}, "request_id": {"id": 0}})
        print_answers(prediction, details="all")


def parse_cmd():
    desc = 'generate documentstore for macro dataset...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-m', type=str, default='dev', dest='mode', help='dev or train', choices=['dev', 'train'])
    args.add_argument('-d', type=str, default='es', dest='ds', help='type of documentstore', choices=['es', 'faiss'])
    args.add_argument('-b', type=int, default=16, dest='bs', help='batch size for DPR')
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    GenDocStoreBase(config)()
