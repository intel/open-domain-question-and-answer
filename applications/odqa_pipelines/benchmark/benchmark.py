from argparse import RawDescriptionHelpFormatter
from typing import List, Dict, Any, Tuple, Optional
from numpy import average
import requests
from multiprocessing import Pool, TimeoutError
import argparse, time, os
import pandas as pd
import numpy as np
DOC_REQUEST = "query"
#API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
#os.environ['NO_PROXY'] = 'localhost'
queries_file = 'test_set_queries.tsv'
qrel_file = 'test_top_k_fixed.tsv'

def load_queries(filename):
    queries = {}
    with open(filename) as fp:
        for q in fp.readlines():
            qid, query_text = q.strip().split("\t")
            query_text = query_text.strip()
            queries[int(qid)] = query_text
    return queries


def load_qrel(filename, queries):
    with open(filename) as fp:
        for q in fp.readlines():
            qid, rel_pid, _ = q.strip().split("\t")
            qid = int(qid)
            rel_pid = int(rel_pid)
            queries[qid] = {"query": queries[qid], "rel_pid": rel_pid}
    return queries

def mrr_at_k(queries, results, at_k=1000):
    mrr = []
    for q in queries:
        relevant_doc = queries[q]["rel_pid"]
        ranks = min(len(results[q]), at_k)
        if relevant_doc in results[q][:ranks]:
            position = results[q][:ranks].index(relevant_doc) + 1
            mrr.append(1.0/position)
        else:
            mrr.append(0.0)
    return np.array(mrr).mean()

def recall_at_k(queries, results, at_k=1000):
    recall = []
    for q in queries:
        relevant_doc = queries[q]["rel_pid"]
        ranks = min(len(results[q]), at_k)
        if relevant_doc in results[q][:ranks]:
            recall.append(1.0)
        else:
            recall.append(0.0)
    return np.array(recall).mean()


def query(query, filters={}, top_k_reader=1000, top_k_ranker=1000, top_k_retriever=1000, pipeline='extractive', mode=0, idx=0, ip_addr='localhost') :
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"http://{ip_addr}:8000/{DOC_REQUEST}"
    ret = 1
    pid = str(os.getpid()) + "_" + str(idx)
    if mode == 0:
        print("ColBERT pipeline performance test!!!");
        params = {"filters": filters, "Retriever": {"top_k": top_k_retriever}, "Ranker": {"top_k": top_k_ranker}, "request_id": {"id": pid}}
    elif mode == 1:
        print("Faiss or Embedding retriever pipeline performance test!!!");
        params = {"filters": filters, "Retriever": {"top_k": top_k_retriever}, "request_id": {"id": pid}}
    else :
        params = {"filters": filters, "Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}, "request_id": {"id": pid}}
    req = {"query": query, "params": params}
    start = time.time()
    response_raw = requests.post(url, json=req)
    interval=time.time() - start
    print(f"{{pid: {pid}}}, {{time: {interval}}}")
    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        ret = 0
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        ret = 0
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    response_ids = []
    answers = response["answers"]
    for answer in answers:
        if answer.get("answer", None):
            results.append(
                {
                    "context": "..." + answer["context"] + "...",
                    "query": answer.get("query", None),
                    "answer": answer.get("answer", None),
                    "source": answer["meta"].get("name", "default"),
                    "relevance": round(answer["score"] * 100, 2),
                    "document": [doc for doc in response["documents"] if doc["id"] == answer["document_id"]][0],
                    "offset_start_in_doc": answer["offsets_in_document"][0]["start"],
                    "_raw": answer,
                }
            )
            response_ids.append(answer["meta"].get("Question_id", None))
        else:
            results.append(
                {
                    "context": None,
                    "query": None,
                    "answer": None,
                    "document": None,
                    "relevance": round(answer["score"] * 100, 2),
                    "_raw": answer,
                }
            )
    return interval, response_ids, ret

def test_accuracy(config):
    print("start processing test set for queries")
    query_results = {}
    with open(queries_file) as input_fd:
        for line in input_fd.readlines():
            qid, query_text = line.strip().split('\t')
            qid = int(qid.strip())
            interval, response_ids, ret = query(query_text, filters={}, top_k_ranker=config.topk, top_k_retriever=config.topk, pipeline='extractive', mode=config.mode, ip_addr=config.ip_address)
            query_results[qid] = response_ids
    print("load qrel and compare")
    test_reference = load_qrel(qrel_file, load_queries(queries_file))
    print(f"MRR @ 3       : {mrr_at_k(test_reference, query_results, at_k=3)}")
    print(f"Recall @ 3    : {recall_at_k(test_reference, query_results, at_k=3)}")

def benchmark(conig, query_idx):
    print(f"Performance benchmark ! Use the default question")
    question="How to get current date in python?"
    interval, response_ids, ret = query(question, filters={}, top_k_ranker=config.topk, top_k_retriever=config.topk, pipeline='extractive', mode=config.mode, idx=query_idx, ip_addr=config.ip_address)
    return interval, ret

def parse_cmd():
    desc = 'multi-process benchmark for haystack...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-p', type=int, default=1, dest='processes', help='How many processes are used for the process pool')
    args.add_argument('-n', type=int, default=1, dest='query_number', help='How many querys will be executed.')
    args.add_argument('-m', type=int, default=0, dest='mode', help='Which pipeline will be tested. 0:colbert; 1:emr or faiss', choices=[0, 1])
    args.add_argument('-b', type=int, default=16, dest='bs', help='batch size for DPR')
    args.add_argument('-a', type=int, default=0, dest='accuracy', help='Is it an accuracy benchmark', choices=[0, 1])
    args.add_argument('-c', type=int, default=0, dest='real_concurrent', help='Use the real concurrent', choices=[0, 1])
    args.add_argument('-t', type=int, default=1000, dest='topk', help='Retriever and Ranker topk')
    args.add_argument('-ip', type=str, default='localhost', dest='ip_address', help='Ip address of backend server')
    return args.parse_args()



if __name__ == '__main__':
    config = parse_cmd()
    if config.accuracy == 0:
        # start 4 worker processes
        result = pd.DataFrame()
        start = time.time()
        if config.real_concurrent == 0 :
            with Pool(processes=config.processes) as pool:

                multiple_results = [pool.apply_async(benchmark, (config, i)) for i in range(config.query_number)]
                for res in multiple_results:
                    interval,ret = res.get()
                    d = {'time':[interval], 'success':[ret]}
                    df = pd.DataFrame(data=d)
                    result = pd.concat([result, df], ignore_index=True)
        else :
            with Pool(processes=config.processes) as pool:
                for num in range(0, int(config.query_number/config.processes)):
                    print(f"concurrent index = {num}")
                    multiple_results = [pool.apply_async(benchmark, (config, num)) for i in range(config.processes)]
                    for res in multiple_results:
                        interval,ret = res.get()
                        d = {'time':[interval], 'success':[ret]}
                        df = pd.DataFrame(data=d)
                        result = pd.concat([result, df], ignore_index=True)    

        total_time = time.time() - start
        result = result.sort_values(by=['time'])
        average_time = result.apply(np.average, axis=0)
        print(average_time)
        print(f"{{query_number: {config.query_number}}}, {{total_time: {total_time}}}, {{fps: {config.processes/average_time.at['time']}}}")
    else:
        print(f"accuracy benchmark")
        test_accuracy(config)

    print("Benchmark Done!")
