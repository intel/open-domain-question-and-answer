from typing import List, Union, Dict

from tqdm.auto import tqdm
import os
from haystack.errors import HaystackError
from haystack.schema import Document, Answer, Span
from haystack.nodes.base import BaseComponent
from ray.data import from_items

class Dataset(BaseComponent):
    """
    This Node is used to convert retrieved documents into predicted answers format.
    It is useful for situations where you are calling a Retriever only pipeline via REST API.
    This ensures that your output is in a compatible format.

    :param progress_bar: Whether to show a progress bar
    """

    outgoing_edges = 1

    def __init__(self, path: str, progress_bar: bool = True):
        super().__init__()
        self.progress_bar = progress_bar
        self.path = path

    def run(self, query: str, documents: List[Document]):  # type: ignore
        pass

    def run_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]]):  # type: ignore
        pass

    def ray_dataset_generator(self) :
        dataset_dirs = os.listdir(self.path)
        dataset_dirs = [ self.path + '/' + x   for x in dataset_dirs if os.path.isdir( self.path + x ) ]
        if len(dataset_dirs) == 0 :
            dataset_dirs = [self.path]
        for dataset_dir in dataset_dirs :
            files = os.listdir(dataset_dir)
            files = [ dataset_dir + '/' + x   for x in os.listdir(dataset_dir) ]
            yield from_items(files)

