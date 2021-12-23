from pathlib import Path

import torch
import numpy as np
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, convert_to_images)


def main(
        biggan_name: str,
        imagenet_classifier_name: str,
        device: torch.device,
        things_images_path: str,
        things_supplementary_path: str,
        results_path: str,
        samples_per_concept: int,
):
    model = BigGAN.from_pretrained(biggan_name)
    model.to(device)

    imagenet_classifier_path = Path(things_supplementary_path) / "ImageNet-Classification" / imagenet_classifier_name
    class_vector_path = imagenet_classifier_path / "concept_averages" / "one_hot.csv"
    concept_imagenet_classes = np.loadtxt(str(class_vector_path), delimiter=",")[:, 1:]

    out_path = Path(results_path) / "Things-ImageNet-Concept-BigGAN-Samples" / biggan_name / imagenet_classifier_name
    out_path.mkdir(exist_ok=True, parents=True)

    for i, class_vector in enumerate(list(concept_imagenet_classes)):
        outputs = []
        for _ in range(samples_per_concept):
            truncation = 0.4
            class_vector = concept_imagenet_classes[i][None]
            noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)

            noise_vector = torch.from_numpy(noise_vector)
            class_vector = torch.from_numpy(class_vector).float()

            noise_vector = noise_vector.to(device)
            class_vector = class_vector.to(device)

            with torch.no_grad():
                outputs.append(model(noise_vector, class_vector, truncation))

        output = torch.cat(outputs)
        output = output.to('cpu')

        images = convert_to_images(output)
        for j, image in enumerate(images):
            image.save(out_path / f"concept{i:04}-sample{j:03}.png", 'png')


if __name__ == "__main__":
    main(
        biggan_name='biggan-deep-256',
        imagenet_classifier_name='resnet152',
        device=torch.device('cuda'),
        things_images_path="X:\\Datasets\\EEG\\Things-concepts-and-images\\",
        things_supplementary_path="X:\\Datasets\\EEG\\Things-supplementary\\",
        results_path="X:\\Results\\Neurophysical-Data-Decoding\\",
        samples_per_concept=12,
    )
