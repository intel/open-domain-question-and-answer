from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import time
import logging
import base64
import torch
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.ranker import BaseRanker
from haystack.schema import Document
from torch.nn import DataParallel

from colbert.modeling.checkpoint import Checkpoint
from colbert.infra.config import ColBERTConfig
from haystack import config

logger = logging.getLogger(__name__)

def decode(arr_str, dtype=None) -> np.ndarray:
        if dtype is None:
            # assume we're using bigendian ordered 31 bit floats
            dtype = np.dtype(np.float32)
            dtype = dtype.newbyteorder('>')  # use big-endian byte ordering
        return np.frombuffer(base64.b64decode(arr_str), dtype=dtype)

class ColBERTRanker(BaseRanker):
    def __init__(
        self,
        model_path: Union[str, Path],
        top_k: int = 10,
        query_maxlen: int = 100,
        doc_maxlen: int = 120,
        dim: int = 128,
        similarity: str = "l2",
        mask_punctuation: bool = True,
        rank: int = -1,
        amp: bool = False,
        use_gpu: bool = True,
        batch_size: int = 1,
        devices: Optional[List[Union[int, str, torch.device]]] = None,
    ):


        self.top_k = top_k
        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        logger.info(f"devices ='{self.devices}'")
        self.batch_size = batch_size

        base_config = ColBERTConfig(
                doc_maxlen=doc_maxlen,
                query_maxlen=query_maxlen,
                dim = dim,
                similarity = similarity,
                amp = amp,
                bsize = batch_size
            )
        checkpoint_config = ColBERTConfig.load_from_checkpoint(model_path)
        final_config = ColBERTConfig.from_existing(checkpoint_config, base_config)
        self.transformer_model = Checkpoint(model_path, colbert_config=final_config)
        self.transformer_model.eval()
        self.transformer_model.to(str(self.devices[0]))

        if len(self.devices) > 1:
            self.model = DataParallel(self.transformer_model.colbert, device_ids=self.devices)
    
    def _encode_query(self, query_text, batch_size=1, to_cpu=True):
        return self.transformer_model.queryFromText([query_text], bsize=batch_size, to_cpu=to_cpu)

    def _encode_doc(self, doc_text: str, batch_size=1, to_cpu=True):
        return self.transformer_model.docFromText([doc_text], bsize=batch_size, to_cpu=to_cpu)

    def _encode_multiple_docs(self, docs: List[str], batch_size=1, to_cpu=True):
        return self.transformer_model.docFromText(docs, bsize=batch_size, to_cpu=to_cpu)

    def calc_score(self, Q, D, mask=None):
        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        return scores.values.sum(-1).cpu()

    def predict(
        self, query: str, documents: List[Document], top_k: Optional[int] = None, to_cpu=True
    ):
        if top_k is None:
            top_k = self.top_k
        start = time.time()
        Q = self._encode_query(query, batch_size=1, to_cpu=to_cpu)
        time1 = time.time()
        if config.COLBERT_OPT == "False" :
            docs_str = [d.content for d in documents]
            docs = self._encode_multiple_docs(docs_str, batch_size=self.batch_size, to_cpu=to_cpu)
            docs = torch.stack(list(docs[0]), dim=0)
            time2 = time.time()
            scores = self.calc_score(Q.permute(0, 2, 1), docs)
        else:
            docs_embedding = [d.to_dict()["meta"]["colbert_emb"] for d in documents]
            docs_tensor = [torch.reshape(torch.tensor(decode(embedding).tolist(), dtype=torch.float32), (-1, 128))  for embedding in docs_embedding]
            if config.IPEX_BF16 == "True":
                docs_tensor=[t.to(dtype=torch.bfloat16) for t in docs_tensor]

            docs_tensor = torch.nn.utils.rnn.pad_sequence(docs_tensor, batch_first=True)
            time2 = time.time()
            scores = self.calc_score(Q.permute(0, 2, 1), docs_tensor)
        time3 = time.time()
        # rank documents according to scores
        sorted_scores_and_documents = sorted(
            zip(scores, documents),
            key=lambda similarity_document_tuple: similarity_document_tuple[0],
            reverse=True,
        )

        sorted_documents = []
        for _, doc in sorted_scores_and_documents :
            del doc.to_dict()["meta"]['colbert_emb']
            sorted_documents.append(doc)

        time4 = time.time()
        return sorted_documents[:top_k]

    def predict_batch(
        self,
        query_doc_list: List[dict],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        raise NotImplementedError


if __name__ == "__main__":
    model_dir = "/nfs/pdx/home/pizsak/store/nosnap/colbert/colbert-so-train/train.py/2021-11-14_09.50.16/checkpoints/"
    model_file = "colbert.dnn"
    model_fullpath = model_dir + model_file
    colbert_ranker = ColBERTRanker(
        model_path=model_fullpath,
        top_k=100,
        amp=True,
    )
    q = "line break tag in html"
    docs = [
        "What is the line break tag in HTML?",
        "how to do line breaks on web pages",
        "what is a paragraph in html?",
    ]
