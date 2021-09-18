from pathlib import Path
from typing import Callable, Optional
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class ThingsDataset(Dataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        self.concepts = pd.read_csv(self.root / "Main" / "things_concepts.tsv",  sep='\t')

        variables_path = self.root / "Variables"
        self.image_concept_ids = np.loadtxt(str(variables_path / "image_concept_index.csv"), dtype=int)
        self.image_paths = np.loadtxt(str(variables_path / "image_paths.csv"), dtype=str)
        self.unique_id = np.loadtxt(str(variables_path / "unique_id.csv"), dtype=str)
        self.wordnet_id = np.loadtxt(str(variables_path / "wordnet_id.csv"), dtype=str)

        self.images = []
        for i in range(len(self.image_concept_ids)):
            concept_id = self.image_concept_ids[i] - 1
            unique_id = self.unique_id[concept_id]
            image_path = self.image_paths[i]
            self.images.append({
                "concept_id": concept_id,
                "unique_id": unique_id,
                "path": str(self.root / "Main" / image_path)
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        image = deepcopy(image)

        with open(image['path'], 'rb') as f:
            data = Image.open(f)
            data = data.convert('RGB')

        if self.transform:
            data = self.transform(data)
        image['data'] = data

        return image

