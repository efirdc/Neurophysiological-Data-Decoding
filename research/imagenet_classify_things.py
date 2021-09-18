from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T

from .things_dataset import ThingsDataset
from .imagenet_classes import imagenet_classes
from pipeline.compact_json_encoder import CompactJSONEncoder


def main(
        model: nn.Module,
        model_name: str,
        device: torch.device,
        things_images_path: str,
        things_supplementary_path: str,
        batch_size: int
):
    model.eval()

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])

    things_dataset = ThingsDataset(
        root=things_images_path,
        transform=transform
    )

    model = model.to(device)
    dataloader = DataLoader(things_dataset, batch_size=batch_size)

    predictions = []
    for iteration, batch in enumerate(dataloader):
        i = iteration * batch_size
        if iteration % 10 == 0:
            print(f"Completed {i}/{len(things_dataset)} classifications.")

        data = batch['data']

        x = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(data)
        x = x.to(device)

        with torch.no_grad():
            y_pred = model(x)
        y_pred = F.softmax(y_pred, dim=1)
        predictions.append(y_pred)

    one_hot = torch.cat(predictions).cpu().numpy()
    out_path = Path(things_supplementary_path) / "ImageNet-Classification" / model_name
    output_results(
        one_hot,
        out_path=out_path / "all_images",
        concepts=[image['unique_id'] for image in things_dataset.images]
    )

    concepts_one_hot_grouped = {}
    for i, one_hot_image in enumerate(list(one_hot)):
        image = things_dataset.images[i]
        concept = image['unique_id']
        if concept not in concepts_one_hot_grouped:
            concepts_one_hot_grouped[concept] = []
        concepts_one_hot_grouped[concept].append(one_hot_image)

    output_results(
        one_hot=np.stack([np.stack(one_hot_group).mean(axis=0) for one_hot_group in concepts_one_hot_grouped.values()]),
        out_path=out_path / "concept_averages",
        concepts=list(concepts_one_hot_grouped.keys())
    )


def output_results(
        one_hot: np.array,
        out_path: Path,
        concepts: Sequence[str]
):
    categorical = np.argmax(one_hot, axis=1)

    out_path.mkdir(exist_ok=True, parents=True)

    pd.DataFrame(one_hot).to_csv(out_path / f"one_hot.csv", header=False)
    pd.DataFrame(categorical).to_csv(out_path / f"categorical.csv", header=False)

    k = 5
    args = np.argsort(one_hot, axis=1)[:, ::-1]
    k_biggest_args = args[:, :k]
    classification = {}
    for i, arg in enumerate(k_biggest_args):
        concept = concepts[i]
        probabilities = [float(p) for p in list(one_hot[i, arg])]
        classes = [imagenet_classes[a] for a in list(arg)]
        classes = [c.split(',')[0] for c in classes]
        classification[i] = {"concept": concept, "imagenet_classification": tuple(zip(classes, probabilities))}

    json_encoder = CompactJSONEncoder(indent=2)
    with (out_path / "classification.json").open(mode="w") as f:
        out_str = json_encoder.encode(classification)
        f.write(out_str)


if __name__ == "__main__":
    main(
        model=models.resnet152(pretrained=True),
        model_name='resetnet152',
        device=torch.device('cuda'),
        things_images_path="X:\\Datasets\\EEG\\Things-concepts-and-images\\",
        things_supplementary_path="X:\\Datasets\\EEG\\Things-supplementary\\",
        batch_size=8,
    )
