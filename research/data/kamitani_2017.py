from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
from copy import deepcopy
import json
import io

import torch
import numpy as np
from torch.utils.data import Dataset
import torchio as tio
import pandas as pd
from PIL import Image
from bids import BIDSLayout
import h5py


class Kamitani2017H5Preprocessed(Dataset):
    def __init__(
            self,
            kamitani_preprocessed_root: str,
            subjects: Sequence[str],
            func_sessions: Sequence[str],
            features_path: str = None,
            fix_stimulus_ids: bool = True,
    ):
        self.root = Path(kamitani_preprocessed_root)
        file_paths = list(self.root.iterdir())
        file_names = [f.name for f in file_paths]

        func_session_map = {
            'natural_test': 'ImageNetTest',
            'natural_training': 'ImageNetTraining',
            'imagery': 'Imagery',
        }

        self.stimulus_info = {
            func_session: pd.read_csv(self.root / f'stimulus_{func_session_map[func_session]}.tsv',
                                      sep='\t', index_col=1, names=['stimulus_id', 'bad_id', '_1', '_2'])
            for func_session in ('natural_test', 'natural_training')
        }
        self.stimulus_info = {
            func_session: df.to_dict()['stimulus_id']
            for func_session, df in self.stimulus_info.items()
        }
        if fix_stimulus_ids:
            for key_map in self.stimulus_info.values():
                for i, stimulus_id in key_map.items():
                    if stimulus_id[0] != 'n':
                        continue
                    wordnet_id, dataset_id = stimulus_id.split('_')
                    key_map[i] = f'{int(wordnet_id[1:])}.{int(dataset_id):06}'

        self.subjects = {subject_name: {} for subject_name in subjects}
        for subject in subjects:
            for func_session in func_sessions:
                session_name = func_session_map[func_session]

                h5_file_name = f'{subject}_{session_name}.h5'
                self.subjects[subject][func_session] = h5py.File(self.root / h5_file_name, 'r')

        self.f_features = None
        if features_path:
            self.f_features = h5py.File(Path(features_path), 'r')
            first_key = list(self.f_features)[0]
            self.feature_shapes = {k.replace('.', '_'): v.shape for k, v in self.f_features[first_key].items()}

    def get_data(self, brain_keys: Sequence[str], feature_keys: Sequence[str] = None):
        non_mask_keys = ('voxel_x', 'voxel_y', 'voxel_z', 'voxel_i', 'voxel_j', 'voxel_k')

        out = {}
        for subject in self.subjects.keys():
            out[subject] = {}
            for func_session in self.subjects[subject].keys():
                out[subject][func_session] = {}

                f = self.subjects[subject][func_session]
                data = f['dataset'][:]
                metadata_keys = {key.decode('utf-8'): i for i, key in enumerate(f['metadata/key'][:])}
                metadata_values = f['metadata/value'][:]
                for metadata_key, i in metadata_keys.items():
                    if metadata_key not in brain_keys:
                        continue
                    if metadata_key in non_mask_keys:
                        out[subject][func_session][metadata_key] = metadata_values[i]
                    else:
                        metadata_mask = metadata_values[i] == 1.
                        out[subject][func_session][metadata_key] = data[:, metadata_mask]

                if self.f_features:
                    image_index_mask = metadata_values[metadata_keys['stimulus_number']] == 1
                    stimulus_image_keys = data[:, image_index_mask]
                    stimulus_ids = [self.stimulus_info[func_session][float(i)] for i in stimulus_image_keys]
                    for feature_key in feature_keys:
                        out[subject][func_session][feature_key] = np.stack([
                            self.f_features[stimulus_id][feature_key][:]
                            for stimulus_id in stimulus_ids
                        ])

        return out


if __name__ == '__main__':
    root = "X:\\Datasets\\Generic-Object-Decoding\\"
    dataset = Kamitani2017H5Preprocessed(root,
                                         subjects=['Subject1',],
                                         func_sessions=['natural_training',],
                                         )
    X_key = 'VoxelData'
    Y_key = 'visual.layer4.7.bn3'
    data = dataset.get_data(brain_keys=[X_key, 'image_index'], feature_keys=[Y_key])