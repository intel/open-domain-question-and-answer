import ujson
import torch
from colbert.modeling.colbert import ColBERT

from colbert.parameters import DEVICE
from colbert.utils.utils import load_checkpoint, print_message
from colbert.utils.runs import Run
from haystack import config
import logging
logger = logging.getLogger(__name__)

def load_model(args, device, do_print=True):
    colbert = ColBERT.from_pretrained(
        "bert-base-uncased",
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        similarity_metric=args.similarity,
        mask_punctuation=args.mask_punctuation,
    )
    colbert = colbert.to(device)

    print_message("#> Loading model checkpoint.", condition=do_print)
    if config.IS_DICT_CHECKPOINT == 'False' :
        checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)
    else :
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        colbert.load_state_dict(checkpoint)
    colbert.eval()

    if config.ENABLE_IPEX == 'True' :
        import intel_extension_for_pytorch as ipex
        colbert = ipex.optimize(colbert)
        if config.IPEX_BF16 == 'True' :
            logger.info("ipex BF16 optimization is enbaled!")
            colbert = ipex.optimize(colbert, dtype=torch.bfloat16)
        else :
            logger.info("ipex FP32 optimization is enbaled!")
            colbert = ipex.optimize(colbert)
    return colbert, checkpoint


def load_colbert(args, device,  do_print=True):
    colbert, checkpoint = load_model(args, device, do_print)

    # TODO: If the parameters below were not specified on the command line, their *checkpoint* values should be used.
    # I.e., not their purely (i.e., training) default values.

    for k in ["query_maxlen", "doc_maxlen", "dim", "similarity", "amp"]:
        if "arguments" in checkpoint and hasattr(args, k):
            if k in checkpoint["arguments"] and checkpoint["arguments"][k] != getattr(args, k):
                a, b = checkpoint["arguments"][k], getattr(args, k)
                print(f"Got checkpoint['arguments']['{k}'] != args.{k} (i.e., {a} != {b})")

    if "arguments" in checkpoint:
        if args.rank < 1:
            print(ujson.dumps(checkpoint["arguments"], indent=4))

    if do_print:
        print("\n")

    return colbert, checkpoint
