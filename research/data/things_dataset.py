from pathlib import Path
from typing import Callable, Optional
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class ThingsDataset(Dataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            supplementary_path: Optional[str] = None,
            latent_name: Optional[str] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        self.concepts = pd.read_csv(self.root / "Main" / "things_concepts.tsv",  sep='\t')

        variables_path = self.root / "Variables"
        self.image_concept_ids = np.loadtxt(str(variables_path / "image_concept_index.csv"), dtype=int)
        self.image_paths = np.loadtxt(str(variables_path / "image_paths.csv"), dtype=str)
        self.unique_image_names = [Path(path).stem for path in self.image_paths]
        self.unique_concept_names = np.loadtxt(str(variables_path / "unique_id.csv"), dtype=str)
        self.wordnet_id = np.loadtxt(str(variables_path / "wordnet_id.csv"), dtype=str)

        self.images = []
        for i in range(len(self.image_concept_ids)):
            concept_id = self.image_concept_ids[i] - 1
            unique_concept_names = self.unique_concept_names[concept_id]
            image_path = self.image_paths[i]
            unique_image_name = self.unique_image_names[i]
            self.images.append({
                "concept_id": concept_id,
                "unique_concept_name": unique_concept_names,
                "path": str(self.root / "Main" / image_path),
                "unique_image_name": unique_image_name,
                'in_image_net': unique_image_name.endswith("n"),
            })
        self.images_map = {}
        for image in self.images:
            if image['unique_image_name'] in self.images_map:
                other_image = self.images_map[image['unique_image_name']]
                raise RuntimeError(f"Images have the same name: {other_image['path']} and {image['path']}")
            self.images_map[image['unique_image_name']] = image

        self.concept_image_map = {unique_concept_name: [] for unique_concept_name in self.unique_concept_names}
        for image in self.images:
            self.concept_image_map[image['unique_concept_name']].append(image)

        if supplementary_path is not None:
            self.supplementary_path = Path(supplementary_path)

            if latent_name is not None:
                self.latent_name = latent_name
                latents_path = self.supplementary_path / "Latents" / latent_name
                for image in self.images:
                    image['latents'] = {}
                for latent_file_path in latents_path.iterdir():
                    latents = np.load(latent_file_path)
                    latent_name = latent_file_path.stem
                    for latent, image in zip(list(latents), self.images):
                        image['latents'][latent_name] = latent

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            image = self.images[idx]
        elif isinstance(idx, str):
            image = self.images_map[idx]
        else:
            raise ValueError(f"Only string and integer indexing supporting. Got {idx} of type {type(idx)}.")
        image = self.load(image)
        return image

    def load(self, image):
        image = deepcopy(image)

        with open(image['path'], 'rb') as f:
            data = Image.open(f)
            data = data.convert('RGB')

        if self.transform:
            data = self.transform(data)
        image['data'] = data
        return image


