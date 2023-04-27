import time
from typing import Dict, Any, Tuple, Sequence, Callable, Optional
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import h5py

from research.metrics.metrics import (
    r2_score,
    pearsonr,
    cosine_distance,
    mean_squared_distance,
    contrastive_score,
    two_versus_two,
    evaluate_decoding
)
from pipeline.utils import merge_dicts, nested_insert, get_data_iterator


def eval_sd_decoding(nsd_path: str):
    results_path = Path(nsd_path) / 'derivatives/decoded_features'

    group_name = 'group-4'
    fold = 'val'

