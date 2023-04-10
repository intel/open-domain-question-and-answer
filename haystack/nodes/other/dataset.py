from typing import List, Union, Optional
from abc import abstractmethod
from haystack.schema import Document
from haystack.nodes.base import BaseComponent
import ray,os

class Dataset(BaseComponent):
    """
    This Node is used to convert dataset into haystack Documents or files path format.
    It is useful for integrated the different dataset into haystack indexing pipeline.
    It uses the ray distributed processing for the dataset. 
    This ensures that your output is in a compatible format.
    """

    outgoing_edges = 1

    def __init__(self, batch_size: Optional[int]):
        super().__init__()
        self.dataset = None
        self.batch_size = batch_size

    @abstractmethod
    def convert(self) -> ray.data.Dataset:
        """
        Convert a dataset to ray.data.Dataset of Haystack Documents or files path.
        """
        pass
 
    def run(self):  # type: ignore
        # conversion from dataset-> Documents or files path
        self.dataset = self.convert()
        enable_sample = os.getenv('ENABLE_SAMPLING_LIMIT', default="0")
        if enable_sample == "1" :
            self.dataset = self.dataset.limit(500)
        return {}, "output_1"

    def run_batch(self):  # type: ignore
        return self.run()

    def dataset_batched_generator(self) :
      """
      Generator to generate the batched haystack Documents or batched files path
      """
      return self.dataset.iter_batches(batch_size=self.batch_size)