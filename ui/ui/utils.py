from typing import List, Dict, Any, Tuple, Optional

import os
import logging
from time import sleep

import requests
import streamlit as st
import yaml

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "")
STATUS = "initialized"
HS_VERSION = "hs_version"
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"


def haystack_is_ready():
    """
    Used to show the "Haystack is loading..." message
    """
    url = f"{API_ENDPOINT}/{STATUS}"
    try:
        if requests.get(url).status_code < 400:
            return True
    except Exception as e:
        logging.exception(e)
        sleep(1)  # To avoid spamming a non-existing endpoint at startup
    return False


@st.cache
def haystack_version():
    """
    Get the Haystack version from the REST API
    """
    url = f"{API_ENDPOINT}/{HS_VERSION}"
    return requests.get(url, timeout=0.1).json()["hs_version"]


def query(query, filters={}, top_k_params = {}) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {"filters": filters}
    params.update(top_k_params)
    print(f"params = {params}")
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    answers = response["answers"]
    for answer in answers:
        answer_score = answer["score"]
        answer_score = round(answer_score * 100, 2) if answer_score < 1.0 else round(answer_score, 2)
        if answer.get("answer", None):
            results.append(
                {
                    "context": "..." + answer["context"] + "...",
                    "query": answer.get("query", None),
                    "answer": answer.get("answer", None),
                    "source": answer["meta"].get("name", "default"),
                    "relevance": answer_score,
                    "document": [doc for doc in response["documents"] if doc["id"] == answer["document_id"]][0],
                    "offset_start_in_doc": answer["offsets_in_document"][0]["start"],
                    "_raw": answer,
                }
            )
        else:
            results.append(
                {
                    "context": None,
                    "query": None,
                    "answer": None,
                    "document": None,
                    "relevance": answer_score,
                    "_raw": answer,
                }
            )
    return results, response


def send_feedback(query, answer_obj, is_correct_answer, is_correct_document, document) -> None:
    """
    Send a feedback (label) to the REST API
    """
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "query": query,
        "document": document,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document,
        "origin": "user-feedback",
        "answer": answer_obj,
    }
    response_raw = requests.post(url, json=req)
    if response_raw.status_code >= 400:
        raise ValueError(f"An error was returned [code {response_raw.status_code}]: {response_raw.json()}")


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response = requests.post(url, files=files).json()
    return response

def load_config(file):
    dataset_config = None
    pipeline_config = None
    with open(file, 'r', encoding='utf-8') as config_file:
        content = config_file.read()
        config = yaml.safe_load(content)
        if PIPELINE_PATH != "":
            pipeline_name = PIPELINE_PATH.split("/")[-1]
            print(f'pipeline_name={pipeline_name}')
            for pipeline in config["pipelines"] :
                if pipeline["name"] == pipeline_name :
                    pipeline_config = pipeline
        else:
            return None, None
        dataset_config = config["dataset"]

    return dataset_config, pipeline_config




def get_backlink(result) -> Tuple[Optional[str], Optional[str]]:
    if result.get("document", None):
        doc = result["document"]
        if isinstance(doc, dict):
            if doc.get("meta", None):
                if isinstance(doc["meta"], dict):
                    if doc["meta"].get("url", None) and doc["meta"].get("title", None):
                        return doc["meta"]["url"], doc["meta"]["title"]
    return None, None
