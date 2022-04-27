from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from random import Random
from typing import Callable, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, Subset
import pandas as pd
import h5py
import nibabel as nib
from einops import rearrange

from pipeline.utils import index_unsorted

BETAS_SCALE = 300
BETAS_PER_SESSION = 750


class NaturalScenesDataset:
    def __init__(
            self,
            dataset_path: str,
            resolution_name: str = 'func1pt8mm',
            preprocess_name: str = 'betas_fithrf_GLMdenoise_RR',
    ):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path / 'derivatives'
        self.ppdata_path = self.dataset_path / 'nsddata' / 'ppdata'

        self.subjects = {f'subj0{i}': {} for i in range(1, 9)}
        for subject_name, subject_data in self.subjects.items():
            responses_file_path = self.ppdata_path / subject_name / 'behav' / 'responses.tsv'
            subject_data['responses'] = pd.read_csv(responses_file_path, sep='\t',)

            # The last 3 sessions are currently held-out for the algonauts challenge
            # remove them for now.
            session_ids = subject_data['responses']['SESSION']
            held_out_mask = session_ids > (np.max(session_ids) - 3)
            subject_data['responses'] = subject_data['responses'][~held_out_mask]

            subject_betas_path = self.derivatives_path / 'betas' / subject_name / resolution_name / preprocess_name
            num_sessions = np.max(subject_data['responses']['SESSION'])

            subject_data['betas'] = h5py.File(subject_betas_path / f'betas_sessions.hdf5', 'r')

            func_path = self.ppdata_path / subject_name / resolution_name
            roi_path = func_path / 'roi'
            roi_paths = {
                **{name: func_path / f'{name}.nii.gz'
                   for name in ('brainmask', 'aseg', 'hippoSfLabels')},
                **{name: roi_path / f'{name}.nii.gz'
                   for name in ('corticalsulc', 'floc-bodies', 'floc-faces', 'floc-places', 'floc-words',
                                'HCP_MMP1', 'Kastner2015', 'MTL', 'nsdgeneral', 'prf-eccrois', 'pip install pysurfer',
                                'streams', 'thalamus')},
            }

            subject_data['roi_paths'] = roi_paths
            label_name_path = self.dataset_path / 'nsddata' / 'freesurfer' / subject_name / 'label'
            ctab_files = [p for p in label_name_path.iterdir() if p.suffix == '.ctab']
            subject_data['roi_label_names'] = label_names = {}
            for roi_name in roi_paths.keys():
                for ctab_file in ctab_files:
                    if ctab_file.name.startswith(roi_name):
                        with open(ctab_file) as f:
                            lines = [line.strip().split(' ') for line in f.readlines()]
                            label_names[roi_name] = {int(line[0]): line[1] for line in lines}
                        label_names[roi_name][-1] = 'Unlabeled'

    def get_affine(self, subject_name):
        return nib.load(self.subjects[subject_name]['roi_paths']['brainmask']).affine

    def load_roi(
            self,
            subject_name,
            roi_name,
    ):
        subject = self.subjects[subject_name]
        image = nib.load(subject['roi_paths'][roi_name]).get_fdata()
        image = image.astype(int)
        label_names = {}
        if roi_name in subject['roi_label_names']:
            label_names = subject['roi_label_names'][roi_name]
        return image, label_names

    def load_betas(
            self,
            subject_name: str,
            voxel_selection_path: str,
            voxel_selection_key: Sequence[str],
            num_voxels: Optional[int] = None,
            threshold: Optional[float] = None,
            return_volume_indices: bool = False,
    ):
        subject_betas = self.subjects[subject_name]['betas']
        voxel_selection_file = h5py.File(self.dataset_path / voxel_selection_path, 'r')
        key = f'{subject_name}/{voxel_selection_key}'
        voxel_selection_map = voxel_selection_file[key][:].flatten()

        if num_voxels is not None:
            indices_flat = voxel_selection_map[:num_voxels]
        elif threshold is not None:
            indices_flat = np.where(voxel_selection_map > threshold)[0]
        else:
            raise ValueError()

        betas = np.stack([
            subject_betas['betas'][:, i]
            for i in indices_flat
        ], axis=1)
        betas = torch.from_numpy(betas).float() / BETAS_SCALE

        N, V = betas.shape
        betas = betas.reshape(BETAS_PER_SESSION, N // BETAS_PER_SESSION, V)
        betas_mean = subject_betas['mean'][:][:, indices_flat]
        betas_std = subject_betas['std'][:][:, indices_flat]

        betas = (betas - betas_mean) / betas_std
        betas = betas.reshape(N, V)
        betas = TensorDataset(betas)

        if return_volume_indices:
            volume_indices = subject_betas['indices'][:][indices_flat]
            return betas, volume_indices
        else:
            return betas

    def load_stimulus(
            self,
            subject_name: str,
            stimulus_path: str = 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5',
            stimulus_key: str = 'imgBrick',
            delay_loading: bool = False,
    ):
        stimulus_file = h5py.File(self.dataset_path / stimulus_path, 'r')
        stimulus = stimulus_file[stimulus_key]

        responses = self.subjects[subject_name]['responses']
        response_stimulus_ids = responses['73KID'].to_numpy() - 1

        if delay_loading:
            return StimulusDataset(stimulus, response_stimulus_ids)
        else:
            stimulus_data = index_unsorted(stimulus, response_stimulus_ids)
            return TensorDataset(stimulus_data)

    def get_split(
            self,
            subject_name: str,
            split_name: str,
    ):
        split = h5py.File(self.derivatives_path / 'data_splits' / f'{split_name}.hdf5')
        subject_split = split[subject_name]

        test_mask = subject_split['test_response_mask'][:].astype(bool)
        val_mask = subject_split['validation_response_mask'][:].astype(bool)
        train_mask = ~(test_mask | val_mask)
        return train_mask, val_mask, test_mask

    def apply_subject_split(
            self,
            dataset: Dataset,
            subject_name: str,
            split_name: str,
    ):
        train_mask, val_mask, test_mask = self.get_split(subject_name, split_name)
        train_dataset = Subset(dataset, np.where(train_mask)[0])
        val_dataset = Subset(dataset, np.where(val_mask)[0])
        test_dataset = Subset(dataset, np.where(test_mask)[0])
        return train_dataset, val_dataset, test_dataset

    def apply_nfold_split(
            self,
            dataset: Dataset,
            num_folds: int,
            select_fold: int,
            seed: int = 0,
    ):
        assert select_fold < num_folds
        fold_ids = [i % num_folds for i in range(len(dataset))]
        Random(seed).shuffle(fold_ids)
        fold_ids = np.array(fold_ids)
        train_dataset = Subset(dataset, np.where(fold_ids != select_fold)[0])
        test_dataset = Subset(dataset, np.where(fold_ids == select_fold)[0])
        return train_dataset, test_dataset

    def combine_nfold_tensors(
            self,
            tensors: Sequence[torch.Tensor],
            num_folds: int,
            seed: int = 0
    ):
        N = sum(tensor.shape[0] for tensor in tensors)
        fold_ids = [i % num_folds for i in range(N)]
        Random(seed).shuffle(fold_ids)
        fold_ids = torch.tensor(fold_ids)

        inverse_subset_ids = torch.argsort(torch.cat([
            torch.where(fold_ids == select_fold)[0]
            for select_fold in range(num_folds)
        ]))

        return torch.cat(tensors)[inverse_subset_ids]

    def reconstruct_volume(
            self,
            subject_name: str,
            values: torch.Tensor,
            indices: torch.Tensor,
            fill_value: Any = 0.,
    ):
        subject_betas = self.subjects[subject_name]['betas']
        volume_shape = tuple(subject_betas['betas'].attrs['spatial_shape'])

        volume = torch.full(volume_shape, fill_value, dtype=values.dtype)
        volume[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        return volume

    def load_data(
            self,
            subject_name: str,
            model_name: str,
            embedding_name: str,
            encoder_name: str,
            split_name: str,
            num_voxels: int,
            normalize_X: bool,
            normalize_Y: bool,
    ):
        encoder_parameters = h5py.File(self.derivatives_path / f'{encoder_name}-parameters.hdf5', 'r')
        sorted_indices_flat = encoder_parameters[subject_name][model_name][embedding_name]['sorted_indices_flat']

        X = np.stack([
            self.subjects[subject_name]['betas']['betas'][:, i]
            for i in sorted_indices_flat[:num_voxels]
        ], axis=1)
        X = torch.from_numpy(X).float() / 300

        model_path = self.derivatives_path / 'stimulus_embeddings' / f'{model_name}-embeddings.hdf5'
        model = h5py.File(model_path, 'r')
        embeddings = model[embedding_name]

        split = h5py.File(self.derivatives_path / 'data_splits' / f'{split_name}.hdf5')
        subject_split = split[subject_name]

        test_response_mask = subject_split['test_response_mask'][:].astype(bool)
        validation_response_mask = subject_split['validation_response_mask'][:].astype(bool)
        training_response_mask = ~(test_response_mask | validation_response_mask)

        responses = self.subjects[subject_name]['responses']
        response_stimulus_ids = responses['73KID'].to_numpy()

        train = NSDFold(X, embeddings, training_response_mask, response_stimulus_ids)
        val = NSDFold(X, embeddings, validation_response_mask, response_stimulus_ids)
        test = NSDFold(X, embeddings, test_response_mask, response_stimulus_ids)

        train.normalize(train, normalize_X, normalize_Y)
        val.normalize(train, normalize_X, normalize_Y)
        test.normalize(train, normalize_X, normalize_Y)

        return train, val, test


class StimulusDataset(Dataset):
    def __init__(self, stimulus: h5py.Dataset, stimulus_ids: torch.Tensor):
        super().__init__()
        self.stimulus = stimulus
        self.stimulus_ids = stimulus_ids

    def __len__(self):
        return self.stimulus_ids.shape[0]

    def __getitem__(self, index):
        stimulus_id = self.stimulus_ids[index]
        stimulus = self.stimulus[stimulus_id]
        return {'data': torch.tensor(stimulus).float(), 'id': stimulus_id}


class KeyDataset(Dataset):
    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        keys = list(self.datasets.keys())
        return len(self.datasets[keys[0]])

    def __getitem__(self, index):
        return {
            key: dataset[index]
            for key, dataset in self.datasets.items()
        }


class NSDFold:
    def __init__(
            self,
            X: torch.Tensor,
            embeddings: h5py.Dataset,
            response_mask: torch.Tensor,
            response_stimulus_ids: torch.Tensor,
    ):
        response_ids = np.where(response_mask)[0]
        stimulus_ids = response_stimulus_ids[response_ids] - 1

        argsort_ids = np.argsort(stimulus_ids)
        unique_results = np.unique(stimulus_ids, return_counts=True)
        unique_stimulus_ids, unique_stimulus_counts = unique_results

        self.X = X[response_ids[argsort_ids]]
        self.X_averaged = torch.stack([
            x.mean(dim=0)
            for x in torch.split(self.X, list(unique_stimulus_counts))
        ])

        self.Y_averaged_ids = unique_stimulus_ids
        self.Y_averaged = embeddings[unique_stimulus_ids]
        self.Y_spatial_shape = self.Y_averaged.shape[1:]
        self.Y_averaged = torch.from_numpy(self.Y_averaged).float()
        self.Y_averaged = self.Y_averaged.flatten(start_dim=1)
        self.Y_ids = torch.repeat_interleave(unique_stimulus_ids, torch.from_numpy(unique_stimulus_counts), dim=0)
        self.Y = torch.repeat_interleave(self.Y_averaged, torch.from_numpy(unique_stimulus_counts), dim=0)

        self.X_mean = self.X.mean(dim=0, keepdims=True)
        self.X_std = self.X.std(dim=0, keepdims=True)

        self.Y_mean = self.Y.mean(dim=0, keepdims=True)
        self.Y_std = self.Y.std(dim=0, keepdims=True)

        self.normalized = False
        self.normalize_params = None

    def normalize(self, other: NSDFold, normalize_X: bool, normalize_Y: bool):
        assert not self.normalized

        if normalize_X:
            self.X = (self.X - other.X_mean) / other.X_std
            self.X_averaged = (self.X_averaged - other.X_mean) / other.X_std
        if normalize_Y:
            self.Y = (self.Y - other.Y_mean) / other.Y_std
            self.Y_averaged = (self.Y_averaged - other.Y_mean) / other.Y_std

        self.normalized = True
        self.normalize_params = (other, normalize_X, normalize_Y)

    def inverse_transform(self, Y: torch.Tensor) -> torch.Tensor:
        other, normalize_X, normalize_Y = self.normalize_params

        if normalize_Y:
            Y = Y * other.Y_std + other.Y_mean
        Y = Y.reshape(Y.shape[0], *self.Y_spatial_shape)

        return Y

    def torch_dataset(self, averaged: bool = False) -> Dataset:
        if averaged:
            return TensorDataset(self.X_averaged, self.Y_averaged)
        return TensorDataset(self.X, self.Y)
