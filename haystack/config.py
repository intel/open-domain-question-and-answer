import os
from pathlib import Path
COLBERT_OPT = os.getenv("COLBERT_OPT", "False")
ENABLE_IPEX = os.getenv("ENABLE_IPEX", "False")
IPEX_BF16 = os.getenv("IPEX_BF16", "False")
IS_DICT_CHECKPOINT = os.getenv("IS_DICT_CHECKPOINT", "False")
