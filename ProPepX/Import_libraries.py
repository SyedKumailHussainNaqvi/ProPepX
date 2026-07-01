import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5EncoderModel
import sys
import os, random, re
import torch

from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import copy
import math
import numpy as np
from datasets import Dataset
import pandas as pd, random, torch
import torch.nn.functional as F 
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import h5py
import hashlib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
