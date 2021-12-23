import os
import sys
from pathlib import Path

import numpy as np
import torch
import mne
import torchvision.transforms as T
import torch.nn.functional as F


from research.data import *


if __name__ == "__main__":
    transform = T.Compose([
        T.Resize(256),
        T.ToTensor(),
    ])

    things_dataset = ThingsDataset(
        root="X:\\Datasets\\EEG\\Things-concepts-and-images\\",
        transform=transform,
        supplementary_path="X:\\Datasets\\EEG\\Things-supplementary\\",
        latent_name="bigbigan-resnet50",
    )

    things_eeg_dataset = ThingsEEG(things_dataset, things_eeg_path="X:\\Datasets\\EEG\\Things-EEG1\\")