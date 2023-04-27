from typing import Sequence, Tuple
from pathlib import Path

import numpy as np
import torch
import fire
from sklearn.model_selection import KFold
from fracridge import FracRidgeRegressorCV
from tqdm import tqdm
import h5py
import nibabel as nib

from research.data.natural_scenes import (
    NaturalScenesDataset,
)
from research.metrics.metrics import (
    r2_score
)
from research.models.regression_torch import frac_ridge_regression_cv


def require_dataset(group, key, tensor):
    if key in group:
        group[key][:] = tensor
    else:
        group[key] = tensor


def main(
        nsd_path: str,
        subject_name: str,
        run_models: Sequence[Tuple[str, str]],
        group_name: str,
        normalize_X: bool = False,
        batch_size: int = 1000,
        max_features: int = 512,
        seed: int = 0,
        permutation_test: bool = False,
):
    print(subject_name)
    nsd = NaturalScenesDataset(nsd_path)
    affine = nsd.get_affine(subject_name)

    train_mask, _, _ = nsd.get_split(subject_name, 'split-01')

    betas_params = dict(
        subject_name=subject_name,
        voxel_selection_path='derivatives/voxel-selection.hdf5',
        voxel_selection_key='nc/value',
        threshold=5.,
        return_volume_indices=True,
        return_tensor_dataset=False,
    )
    betas, betas_indices = nsd.load_betas(**betas_params)
    Y = betas[train_mask]
    num_stimuli, num_voxels = Y.shape

    if permutation_test:
        ids = np.arange(Y.shape[0])
        np.random.shuffle(ids)
        Y = Y[ids]

    for model_name, stimulus_key in run_models:
        print(model_name, stimulus_key)
        stimulus_params = dict(
            subject_name=subject_name,
            #stimulus_path='nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5',
            #stimulus_key='imgBrick',
            stimulus_path=f'derivatives/stimulus_embeddings/{model_name}.hdf5',
            stimulus_key=stimulus_key,
            delay_loading=False,
            return_tensor_dataset=False
        )
        stimulus = nsd.load_stimulus(**stimulus_params)
        X = stimulus[train_mask].astype(np.float32)
        X = X.reshape(num_stimuli, -1)

        if X.shape[1] > max_features:
            np.random.seed(seed)
            choice = np.random.choice(X.shape[1], size=max_features)
            X = X[:, choice]

        if normalize_X:
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-7)

        X = torch.from_numpy(X).cuda()

        coefs = []
        alpha = []
        r2 = []
        fractions = []
        for indices in tqdm(np.array_split(np.arange(num_voxels), num_voxels // batch_size)):
            _coefs, _alpha, _r2, _fractions = frac_ridge_regression_cv(
                X, torch.from_numpy(Y[:, indices]).cuda()
            )
            coefs.append(_coefs.cpu())
            alpha.append(_alpha.cpu())
            r2.append(_r2.cpu())
            fractions.append(_fractions.cpu())
        coefs = torch.cat(coefs, dim=1)
        alpha = torch.cat(alpha)
        r2 = torch.cat(r2)
        fractions = torch.cat(fractions)

        save_file_path = Path(nsd_path) / f'derivatives/encoded_betas/{model_name}/{group_name}'
        save_file_path.mkdir(exist_ok=True, parents=True)

        if not permutation_test:
            with h5py.File(f'{save_file_path}.hdf5', 'a') as f:
                argsort_ids = torch.flip(torch.argsort(r2), [0])
                group = f.require_group(f'{subject_name}/{stimulus_key}')
                require_dataset(group, 'volume_indices', betas_indices[argsort_ids])
                require_dataset(group, 'r2', r2[argsort_ids])
                require_dataset(group, 'coefs', coefs[:, argsort_ids])
                require_dataset(group, 'alpha', alpha[argsort_ids])
                require_dataset(group, 'fractions', fractions[argsort_ids])

        r2_volume = nsd.reconstruct_volume(subject_name, r2, betas_indices)
        fractions_volume = nsd.reconstruct_volume(subject_name, fractions, betas_indices)
        alpha_volume = nsd.reconstruct_volume(subject_name, alpha, betas_indices)

        image_key = '__'.join((subject_name, group_name, model_name, stimulus_key))

        if permutation_test:
            image_key = image_key + '__permutation_test'

        save_file_path.mkdir(exist_ok=True, parents=True)
        nib.save(nib.Nifti1Image(r2_volume.numpy().T, affine), save_file_path / f'{image_key}__r2.nii.gz')
        nib.save(nib.Nifti1Image(fractions_volume.numpy().T, affine), save_file_path / f'{image_key}__frac.nii.gz')
        nib.save(nib.Nifti1Image(alpha_volume.numpy().T, affine), save_file_path / f'{image_key}__alpha.nii.gz')


def clip_fracridge(
        nsd_path: str,
        subject: str = 'all',
        model_name: str = 'ViT-B=32',
        encode_resblocks: bool = False,
):
    group_name = 'fracridge'
    if subject == 'all':
        subjects = [f'subj0{i}' for i in range(1, 9)]
    else:
        subjects = [subject]
    for subject_name in subjects:

        main(
            nsd_path=nsd_path,
            subject_name=subject_name,
            run_models=[(model_name, 'embedding')],
            group_name=group_name,
            normalize_X=False
        )
        if encode_resblocks:
            main(
                nsd_path=nsd_path,
                subject_name=subject_name,
                run_models=[(model_name, f'transformer.resblocks.{i}') for i in range(12)],
                group_name=group_name,
                normalize_X=True
            )


def sd_clip_fracridge(
        nsd_path: str,
        subject: str = 'all',
        permutation_test: bool = False,
        embedding: bool = False,
):
    group_name = 'fracridge'
    if subject == 'all':
        subjects = [f'subj0{i}' for i in range(1, 9)]
    else:
        subjects = [subject]

    if not embedding:
        run_models = [('clip-vit-large-patch14-text', 'embedding_unpooled')]
        normalize_X = True,
    else:
        run_models = [('clip-vit-large-patch14', 'image_embedding'),
                      ('clip-vit-large-patch14', 'text_embedding')]
        normalize_X = False

    print('7')
    for subject_name in subjects:
        main(
            nsd_path=nsd_path,
            subject_name=subject_name,
            run_models=run_models,
            group_name=group_name,
            normalize_X=normalize_X,
            permutation_test=permutation_test,
            max_features=1024,
        )


if __name__ == '__main__':
    fire.Fire({f.__name__: f for f in [main, clip_fracridge, sd_clip_fracridge]})
