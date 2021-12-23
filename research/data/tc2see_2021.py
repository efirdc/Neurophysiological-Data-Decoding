from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
from copy import deepcopy
import json
import io

import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
from numpy.typing import ArrayLike
from torch.utils.data import Dataset
import torchio as tio
import pandas as pd
from PIL import Image
from bids import BIDSLayout
import h5py


class TC2See2021(Dataset):
    def __init__(
            self,
            h5_path: str,
            subjects: Sequence[str],
            cached_preprocessing_name: Optional[str] = None,
            window: Tuple[int, int] = None,
            window_kernel: ArrayLike = None,
            transform: Optional[tio.Transform] = None,
            normalization: Optional[str] = None,
            normalization_steps: Tuple[str] = ('mean', 'std'),
            drop_out_of_window_events: bool = False,
            feature_selection_key: Optional[str] = None,
            feature_selection_path: Optional[str] = None,
            feature_selection_top_k: Optional[int] = None,
            features_path: str = None,
            feature_keys: Optional[Sequence[str]] = None,
            #folds: Optional[Sequence[int]] = None,
            #split: str = 'all',
    ):
        self.window = window
        self.transform = transform
        self.window_kernel = window_kernel
        self.feature_keys = feature_keys

        self.cached_processing = cached_preprocessing_name is not None
        self.cached_processing_name = cached_preprocessing_name

        assert (feature_selection_path is None) == (feature_selection_key is None) == (feature_selection_top_k is None)
        self.feature_selection = feature_selection_path is not None
        self.feature_selection_key = feature_selection_key
        self.feature_selection_path = feature_selection_path
        self.feature_selection_top_k = feature_selection_top_k
        if self.feature_selection:
            self.f_feature_selection = h5py.File(Path(feature_selection_path), 'r')

        assert normalization in (None, 'voxel', 'volume', 'voxel_linear_trend', 'volume_linear_trend')

        self.normalization = normalization
        self.normalization_steps = normalization_steps

        self.f = h5py.File(Path(h5_path), 'r')

        if self.cached_processing:
            self.events = []
            for subject in subjects:

                cache = self.f[f'{subject}/bird_task/{cached_preprocessing_name}']
                onsets = list(cache['onset'][:])
                run_ids = list(cache['run_id'][:])
                stimulus_ids = cache['stimulus_id'][:]
                stimulus_ids = [s.decode('utf-8') for s in stimulus_ids]
                self.events += [
                    {
                        'subject': subject,
                        'subject_event_id': i,
                        'onset': onset,
                        'run_id': run_id,
                        'stimulus_id': stimulus_id
                    }
                    for i, (onset, run_id, stimulus_id) in enumerate(zip(onsets, run_ids, stimulus_ids))
                ]
        else:
            self.runs = []
            for subject in subjects:
                session_runs = list(self.f[f'{subject}/bird_task/runs'].values())
                session_runs.sort(key=lambda dset: int(dset.name.split('/')[-1].split('_')[-1]))
                self.runs += session_runs

            self.run_event_dfs = [
                pd.read_csv(io.StringIO(run.attrs['events']), sep='\t', dtype={'class_id': 'Int64'})
                for run in self.runs
            ]

            self.events = []
            for run_id, run_event_df in enumerate(self.run_event_dfs):
                run_event_df = run_event_df[~run_event_df['class_id'].isnull()]
                events = run_event_df.to_dict("records")
                for event in events:
                    event['run_id'] = run_id
                    event['subject'] = self.runs[run_id].parent.parent.name[1:]
                    stimulus_path = event['stimulus']
                    event['stimulus_id'] = stimulus_path.split('/')[-1][:-4]
                self.events += events

            if drop_out_of_window_events:
                self.events = [
                    event for event in self.events
                    if round(event['tr']) + window[1] < self.runs[event['run_id']]['data'].shape[3]
                    and (round(event['tr']) + window[0] >= 0)
                ]
        # self.events.sort(key=lambda event: event['subject'] + event['tr'])

        # subject_name -> stimulus_id -> event_id_seq
        self.event_map = {}
        for event in self.events:
            subject_name = event['subject']
            stimulus_id = event['stimulus_id']

            if subject_name not in self.event_map:
                self.event_map[subject_name] = {}
            subject_map = self.event_map[subject_name]

            if stimulus_id not in subject_map:
                subject_map[stimulus_id] = []
            event_list = subject_map[stimulus_id]
            event_list.append(event)

        self.stimulus_grouped_events = []
        for stimulus_ids in self.event_map.values():
            for event_list in stimulus_ids.values():
                self.stimulus_grouped_events.append(event_list)

        self.f_features = None
        if features_path:
            self.f_features = h5py.File(Path(features_path), 'r')
            first_key = list(self.f_features)[0]
            self.feature_shapes = {k.replace('.', '_'): v.shape for k, v in self.f_features[first_key].items()}

    def __len__(self):
        return len(self.events)

    def __getitem__(self, event_id):
        event = self.events[event_id]
        event = self.load_event(event)
        return event

    def get_data(self):
        return default_collate([event for event in self])

    def load_event(self, event):
        if self.cached_processing:
            cache = self.f[event['subject']][event['func_session']][self.cached_processing_name]

            if self.feature_selection:
                sorted_indices = self.f_feature_selection[event['subject']][self.feature_selection_key]['sorted_indices']
                top_k = sorted_indices[:, -self.feature_selection_top_k:]
                i, j, k = list(top_k)
                volume_data = cache['data'][event['subject_event_id']]
                event['data'] = volume_data[i, j, k]
            else:
                event['data'] = cache['data'][event['subject_event_id']]
            event['affine'] = cache.attrs['affine']

        else:
            run = self.runs[event['run_id']]

            t_onset = round(event['tr'])
            t_start = max(t_onset + self.window[0], 0)
            t_end = min(t_onset + self.window[1], run['data'].shape[3])
            t = torch.arange(t_start, t_end).int()
            data = torch.from_numpy(run['data'][:, :, :, t.numpy()])

            if self.normalization == 'voxel':
                mean = torch.from_numpy(run['voxel_mean'][:][..., None])
                std = torch.from_numpy(run['voxel_std'][:][..., None])

            elif self.normalization == 'volume':
                mean = torch.from_numpy(run['volume_mean'][t.numpy()][None, None, None])
                std = torch.from_numpy(run['volume_std'][t.numpy()][None, None, None])

            elif self.normalization == 'voxel_linear_trend':
                X = torch.from_numpy(run['voxel_linear_trend'][:])
                std = torch.from_numpy(run['voxel_linear_trend_std'][:][..., None])

                A = torch.zeros_like(data)
                A[:] = t.float()
                A = torch.stack([A, torch.ones_like(A)], dim=-1)
                mean = (A @ X)[..., 0]

            elif self.normalization == 'volume_linear_trend':
                X = torch.from_numpy(run['volume_linear_trend'][:])
                std = torch.from_numpy(run['volume_linear_trend_std'][:])[None, None, None, :]

                A = torch.stack([t, torch.ones_like(t)], dim=-1).float()
                mean = (A @ X)[None, None, None, :]

            if self.normalization is not None:
                if 'mean' in self.normalization_steps:
                    data = data - mean
                if 'std' in self.normalization_steps:
                    data = data / (std + 1e-8)

            if self.window_kernel is not None:
                window_kernel = torch.Tensor(self.window_kernel)[None, None, None, :]
                data = (data * window_kernel).sum(dim=3, keepdims=True)

            data = torch.moveaxis(data, -1, 0)
            bold_image = tio.ScalarImage(
                tensor=data,
                affine=run.attrs['affine']
            )
            if self.transform is not None:
                bold_image = self.transform(bold_image)

            if self.feature_selection:
                sorted_indices = self.f_feature_selection[self.feature_selection_key]['sorted_indices'][:]
                top_k = sorted_indices[:, -self.feature_selection_top_k:]
                i, j, k = list(top_k)
                event['data'] = bold_image['data'][i, j, k]
            else:
                event['data'] = bold_image['data']

            event['affine'] = bold_image['affine']

        if self.f_features is not None:
            stimulus_features = self.f_features[event['stimulus_id']]
            feature_keys = self.feature_keys if self.feature_keys is not None else stimulus_features.keys()
            event['features'] = {
                feature_key.replace('.', '_'): torch.tensor(stimulus_features[feature_key][:]).float()
                for feature_key in feature_keys
            }

        return event.copy()


if __name__ == '__main__':
    ssd_dataset_path = Path('C:\\Datasets\\2021 TC2See fMRI Data\\')
    ssd_derivatives_path = ssd_dataset_path / 'derivatives'
    TC2See2021(h5_path=ssd_derivatives_path / 'TC2See2021.hdf5',
               subjects=['sub-01'],
               window=(-2, 6),)