from pathlib import Path
from typing import Dict

import torch
import numpy as np
from pytorch_pretrained_biggan import convert_to_images
from scipy.stats import truncnorm
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


def truncated_z_sample(batch_size, dim_z, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
    return truncation * values


def one_hot(index, vocab_size):
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output = np.zeros((num, vocab_size), dtype=np.float32)
    output[np.arange(num), index] = 1
    return output


def one_hot_if_needed(label, vocab_size):
    label = np.asarray(label)
    if len(label.shape) <= 1:
        label = one_hot(label, vocab_size)
    assert len(label.shape) == 2
    return label


def sample(
        session: tf.Session,
        noise: np.array,
        label: np.array,
        vocab_size: int,
        module: hub.Module,
        inputs: Dict,
        truncation: float,
        batch_size: int):
    noise = np.asarray(noise, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    num = noise.shape[0]
    if len(label.shape) == 0:
        label = np.asarray([label] * num)
    if label.shape[0] != num:
        raise ValueError(f'Got # noise samples ({noise.shape[0]}) != # label samples ({label.shape[0]})')
    label = one_hot_if_needed(label, vocab_size)
    ims = []
    for batch_start in range(0, num, batch_size):
        s = slice(batch_start, min(num, batch_start + batch_size))
        feed_dict = {inputs['z']: noise[s], inputs['y']: label[s], inputs['truncation']: truncation}
        #ims.append(module(feed_dict))
        ims.append(session.run(module(feed_dict), feed_dict=feed_dict))
    ims = np.concatenate(ims, axis=0)
    assert ims.shape[0] == num
    ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
    ims = np.uint8(ims)
    return ims


def main(
        biggan_name: str,
        module_path: str,
        imagenet_classifier_name: str,
        device: torch.device,
        things_images_path: str,
        things_supplementary_path: str,
        results_path: str,
        samples_per_concept: int,
        noise_seed: int,
):
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    print('Loading BigGAN module from:', module_path)
    module = hub.Module(module_path)
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().items()}
    output = module(inputs)

    print('\nInputs:\n', '\n'.join(f'  {k}: {v}' for k, v in inputs.items()))
    print('\nOutput:', output)

    initializer = tf.global_variables_initializer()
    session = tf.Session()
    session.run(initializer)

    imagenet_classifier_path = Path(things_supplementary_path) / "ImageNet-Classification" / imagenet_classifier_name
    class_vector_path = imagenet_classifier_path / "concept_averages" / "one_hot.csv"
    concept_imagenet_classes = np.loadtxt(str(class_vector_path), delimiter=",")[:, 1:]

    out_path = Path(results_path) / "Things-ImageNet-Concept-BigGAN-Samples" / imagenet_classifier_name / biggan_name
    out_path.mkdir(exist_ok=True, parents=True)

    for i, class_vector in enumerate(list(concept_imagenet_classes)):
        print(i)
        outputs = []
        for _ in range(samples_per_concept):
            truncation = 0.4
            class_vector = concept_imagenet_classes[i][None]
            noise_vector = truncated_z_sample(batch_size=1,
                                              dim_z=int(inputs['z'].shape[1]),
                                              truncation=truncation,
                                              seed=noise_seed)

            output = sample(session=session, noise=noise_vector, label=class_vector, vocab_size=1000,
                            module=module, inputs=inputs, truncation=truncation, batch_size=1)
            outputs.append(output)

        output = torch.cat(outputs)
        output = output.to('cpu')

        images = convert_to_images(output)
        for j, image in enumerate(images):
            image.save(out_path / f"concept{i:04}-sample{j:03}.png", 'png')


if __name__ == "__main__":
    main(
        biggan_name='biggan-deep-256',
        module_path="X:/Models/biggan/biggan-deep-256_1/", # "https://tfhub.dev/deepmind/biggan-deep-256/1",
        imagenet_classifier_name='resnet152',
        device=torch.device('cuda'),
        things_images_path="X:\\Datasets\\EEG\\Things-concepts-and-images\\",
        things_supplementary_path="X:\\Datasets\\EEG\\Things-supplementary\\",
        results_path="X:\\Results\\Neurophysical-Data-Decoding\\",
        samples_per_concept=12,
        noise_seed=0,
    )