from typing import List, Dict
from haystack.schema import Document
from haystack.nodes.other import Dataset
import time
import ray
import modin.pandas as modin_pd
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import os

os.environ["__MODIN_AUTOIMPORT_PANDAS__"] = "1"

def _generate_documents(batch: pd.DataFrame) -> List[Dict]:
    docs_to_index = batch.to_dict(orient="records")
    rest_keys = list(docs_to_index[0].keys())
    rest_keys.remove('text')
    docs = [{'content': d['text'], 'meta': {k: d[k] for k in rest_keys}} for d in docs_to_index]
    documents = [Document.from_dict(d) for d in docs]
    return documents


    
class StackoverflowDataset(Dataset):
    """
    This Node is used to convert stackoverflow dataset into ray.data.Dataset of Haystack Document format.
    """

    outgoing_edges = 1

    def __init__(self,
        question_file: str,
        answer_file: str,
        batch_size: Optional[int] = 4096,
        ) :

        super().__init__(batch_size=batch_size)
        self.question_file = question_file
        self.answer_file = answer_file

    def convert(self) -> ray.data.Dataset:

        print('start to read answers...')
        start = time.time()
        answers = modin_pd.read_csv(self.answer_file, encoding="ISO-8859-1")
        time1 = time.time()
        print(f"read answer csv time ={time1-start}");
        answers.drop(columns=['OwnerUserId', 'CreationDate', 'Id'], inplace=True)
        time2 = time.time()
        print(f"answer drop time ={time2-time1}");
        answers = ray.data.from_modin(answers)
        time3 = time.time()
        print(f"convert pandas to ray df time ={time3-time2}");
        print('start to process answers...')
        top_answers = answers.groupby("ParentId").map_groups(lambda g: g[g['Score'] == g['Score'].max()])
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
        questions = modin_pd.read_csv(self.question_file, encoding="ISO-8859-1")
        time8 = time.time()
        print(f"read questions csv time ={time8-time7}");
        questions.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate', 'Score'], inplace=True)
        time9 = time.time()
        print(f"questions drop time ={time9-time8}");
        questions.rename(columns={"Title": "text", "Body": "Question-html"}, inplace=True)
        time10 = time.time()
        print(f"questions rename time ={time10-time9}");
        qa = questions.merge(top_answers, on='Id')
        time11 = time.time()
        print(f"questions answer merge time ={time11-time10}");
        qa.rename(columns={'Id':'question_id'}, inplace=True)
        time12 = time.time()
        print(f"qa drop time ={time12-time11}");

        # now cleanup if any
        print('Dupplicate entries: {}'.format(qa.duplicated(subset=['text']).sum()))
        #qa.drop_duplicates(inplace=True)
        qa.drop_duplicates(subset=['text'], inplace=True)
        time13 = time.time()
        print(f"qa drop duplicates time ={time13-time12}");
 
        # convert html to plain text
        print('convert html to plain text...')
        qa['question-body'] = qa['Question-html'].apply(lambda x: BeautifulSoup(x).get_text())
        qa['answer'] = qa['TopAnswer-html'].apply(lambda x: BeautifulSoup(x).get_text())
        qa.drop(columns=['Question-html', 'TopAnswer-html'], inplace=True)
        time14 = time.time()
        print(f"convert html time ={time14-time13}");
        dataset = ray.data.from_modin(qa)
        dataset = dataset.map_batches(_generate_documents)
        return dataset

    