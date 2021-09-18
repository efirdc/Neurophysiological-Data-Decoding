from pathlib import Path

import numpy as np
import torchvision.transforms as T
import h5py

from .things_dataset import ThingsDataset


def main(
        imagenet_classifier_name: str,
        things_dataset_path: str,
        things_supplementary_path: str,
        out_file_path: str,
        image_size: int,
        average_concept: bool,
):
    transform = T.Compose([
        T.Resize(image_size),
    ])

    things_dataset = ThingsDataset(
        root=things_dataset_path,
        transform=transform
    )
    N = len(things_dataset)

    imagenet_classifier_path = Path(things_supplementary_path) / "ImageNet-Classification" / imagenet_classifier_name
    class_sub_dir = "concept_averages" if average_concept else "all_images"
    class_vector_path = imagenet_classifier_path / class_sub_dir / "one_hot.csv"
    concept_imagenet_classes = np.loadtxt(str(class_vector_path), delimiter=",")[:, 1:]

    with h5py.File(Path(out_file_path), 'w') as f:
        out_images = f.create_dataset('xtrain', (N, 3, image_size, image_size), dtype='uint8')
        out_labels = f.create_dataset('ytrain', (N, concept_imagenet_classes.shape[1]), dtype='float32')

        image_concept_ids = np.array([image['concept_id'] for image in things_dataset.images])
        out_labels[...] = concept_imagenet_classes[image_concept_ids]
        out_labels.attrs['imagenet_classifier_name'] = imagenet_classifier_name
        out_labels.attrs['average_concept'] = average_concept

        for i, image in enumerate(things_dataset):
            if i % 1000 == 0:
                print(f"Saved image {i}")
            data = np.array(image['data'])
            data = np.moveaxis(data, -1, 0)
            out_images[i] = data


if __name__ == "__main__":
    main(
        things_dataset_path="X:\\Datasets\\EEG\\Things-concepts-and-images\\",
        things_supplementary_path="X:\\Datasets\\EEG\\Things-supplementary\\",
        out_file_path="./things_stimulus01.hdf5",
        imagenet_classifier_name='resnet152',
        image_size=128,
        average_concept=True,
    )
