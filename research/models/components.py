from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)