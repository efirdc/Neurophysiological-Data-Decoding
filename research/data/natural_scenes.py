from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
from copy import deepcopy
import json
import io

import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
from numpy.typing import ArrayLike
from torch.utils.data import Dataset
import torchio as tio
import pandas as pd
from PIL import Image
from bids import BIDSLayout
import h5py

class NaturalScenesDataset(Dataset):