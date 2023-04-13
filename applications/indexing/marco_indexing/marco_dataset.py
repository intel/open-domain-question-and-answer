from typing import List, Union, Optional
from haystack.schema import Document
from haystack.nodes.other import Dataset
import time, ray
import pandas as pd
import modin.pandas as modin_pd
import os
import json
os.environ["__MODIN_AUTOIMPORT_PANDAS__"] = "1"

def _generate_documents(batch: pd.DataFrame) -> List[Document]:
    documents = []
    for _, data in batch.iterrows():
        if isinstance(data['answers'], list) == False:
            continue

        data['answers'] = data['answers'][0]
        if len(str(data['wellFormedAnswers'])) > 2:
            if isinstance(data['wellFormedAnswers'], list) :
                data['answers'] = data['wellFormedAnswers'][0]

        elif "No Answer Present." in data['answers']:
            data['answers'] = data['passages']

        if len(str(data['answers'])) == 0:
            print("no answers, drop the document!")
            continue

        doc = {'content': str(data['query']), 'meta': {'answer': str(data['answers']), 'question_id': str(data['query_id']), 'question_type': str(data['query_type'])}}
        documents.append(Document.from_dict(doc))
    return documents


    
class MarcoDataset(Dataset):
    """
    This Node is used to convert MS Marco dataset into ray.data.Dataset of Haystack Document format.
    """

    outgoing_edges = 1

    def __init__(self,
        file: str,
        batch_size: Optional[int] = 4096,
        ) :

        super().__init__(batch_size=batch_size)
        self.file = file

    def convert(self) -> ray.data.Dataset:
        dataset = modin_pd.read_json(self.file)
        dataset = ray.data.from_modin(dataset)
        start = time.time()
        dataset = dataset.map_batches(_generate_documents)
        cost = time.time() - start
        return dataset
