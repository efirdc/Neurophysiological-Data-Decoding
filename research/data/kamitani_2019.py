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


# session_type, folder_name, task_type
SESSION_INFO = [
    ('imagery', 'imagery', 'imagery'),
    ('artificial', 'perceptionArtificialImage', 'perception'),
    ('letter', 'perceptionLetterImage', 'perception'),
    ('natural_test', 'perceptionNaturalImageTest', 'perception'),
    ('natural_training', 'perceptionNaturalImageTraining', 'perception'),
]

ROI_NAMES = ('LH_FFA', 'LH_hV4', 'LH_HVC', 'LH_LOC', 'LH_PPA', 'LH_V1d',
             'LH_V1v', 'LH_V2d', 'LH_V2v', 'LH_V3d', 'LH_V3v',
             'RH_FFA', 'RH_hV4', 'RH_HVC', 'RH_LOC', 'RH_PPA', 'RH_V1d',
             'RH_V1v', 'RH_V2d', 'RH_V2v', 'RH_V3d', 'RH_V3v')


def infer_num_sessions(subject_path: Path, folder_name: str) -> int:
    return len([p.name for p in subject_path.iterdir() if folder_name in p.name])


def infer_num_runs(session_path: Path) -> int:
    file_names = [p.name for p in session_path.iterdir()]
    run_ids = [int(file_name.split('_')[3][4:6]) for file_name in file_names]
    return max(run_ids)


def fix_stimulus_id(stimulus_id: str, true_stimulus_ids: Sequence[str]):
    if '.' not in stimulus_id:
        return stimulus_id
    wordnet_id, dataset_id = stimulus_id.split('.')
    for i in range(4):
        new_stimulus_id = f'{wordnet_id}.{int(dataset_id) * 10 ** i:06}'
        if new_stimulus_id in true_stimulus_ids:
            return new_stimulus_id
    raise RuntimeError(f"Failed to fix stimulus_id {stimulus_id}")


class Kamitani2019H5Preprocessed(Dataset):
    def __init__(
            self,
            kamitani_preprocessed_root: str,
            subjects: Sequence[str],
            func_sessions: Sequence[str],
            use_v2_files: bool = True,
            features_path: str = None,
            fix_stimulus_ids: bool = True,
    ):
        self.root = Path(kamitani_preprocessed_root)
        file_paths = list(self.root.iterdir())
        file_names = [f.name for f in file_paths]

        func_session_map = {func_session: session_name for func_session, session_name, _ in SESSION_INFO}

        self.stimulus_info = {
            func_session: pd.read_csv(self.root / f'stimulus_{func_session_map[func_session]}.tsv',
                                      sep='\t', index_col=1, names=['stimulus_id', 'index'])
            for func_session in func_sessions
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

                h5_file_name = f'{subject}_{session_name}_VC_v2.h5'
                if not use_v2_files or h5_file_name not in file_names:
                    h5_file_name = f'{subject}_{session_name}_original_VC.h5'

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
                    image_index_mask = metadata_values[metadata_keys['image_index']] == 1
                    stimulus_image_keys = data[:, image_index_mask]
                    stimulus_ids = [self.stimulus_info[func_session][int(i)] for i in stimulus_image_keys]
                    for feature_key in feature_keys:
                        out[subject][func_session][feature_key] = np.stack([
                            self.f_features[stimulus_id][feature_key][:]
                            for stimulus_id in stimulus_ids
                        ])

        return out


class Kamitani2019H5(Dataset):
    def __init__(
            self,
            h5_path: str,
            subjects: Sequence[str],
            func_sessions: Sequence[str],
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
            average_stimulus_repetitions: bool = False,
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

        self.average_stimulus_repetitions = average_stimulus_repetitions

        self.f = h5py.File(Path(h5_path), 'r')

        if self.cached_processing:
            self.events = []
            for subject in subjects:
                for func_session in func_sessions:
                    cache = self.f[f'{subject}/{func_session}/{cached_preprocessing_name}']
                    onsets = list(cache['onset'][:])
                    run_ids = list(cache['run_id'][:])
                    stimulus_ids = cache['stimulus_id'][:]
                    stimulus_ids = [s.decode('utf-8') for s in stimulus_ids]
                    self.events += [
                        {
                            'subject': subject,
                            'func_session': func_session,
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
                for func_session in func_sessions:
                    session_runs = list(self.f[f'{subject}/{func_session}/runs'].values())
                    session_runs.sort(key=lambda dset: int(dset.name.split('/')[-1].split('_')[-1]))
                    self.runs += session_runs

            self.run_event_dfs = [
                pd.read_csv(io.StringIO(run.attrs['events']), index_col=0, dtype={'stimulus_id': str})
                for run in self.runs
            ]

            natural_test_stimulus_ids = self.f.attrs['natural_test_stimulus_ids']
            self.events = []
            for run_id, run_event_df in enumerate(self.run_event_dfs):
                run_event_df = run_event_df[run_event_df['event_type'] == 1]
                run_event_df = run_event_df.filter(items=('onset', 'stimulus_id', 'category_id'), axis='columns')
                events = run_event_df.to_dict("records")
                for event in events:
                    event['run_id'] = run_id
                    event['subject'] = self.runs[run_id].parent.parent.name[1:]
                    if 'category_id' in event:
                        category_id = int(event['category_id'])
                        event['stimulus_id'] = natural_test_stimulus_ids[category_id - 1]
                self.events += events

            if drop_out_of_window_events:
                self.events = [
                    event for event in self.events
                    if event['onset'] // 2 + window[1] < self.runs[event['run_id']]['data'].shape[3]
                ]
        self.events.sort(key=lambda event: event['subject'] + event['stimulus_id'])

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
        if self.average_stimulus_repetitions:
            return len(self.stimulus_grouped_events)
        else:
            return len(self.events)

    def __getitem__(self, event_id):
        if self.average_stimulus_repetitions:
            events = self.stimulus_grouped_events[event_id]
            events = [self.load_event(event) for event in events]
            event = self.average_events(events)
        else:
            event = self.events[event_id]
            event = self.load_event(event)

        return event

    def get_data(self):
        return default_collate([event for event in self])

    def average_events(self, events):
        event = events[0]

        varying_keys = ('subject_event_id', 'onset', 'run_id')
        for key in varying_keys:
            if key not in event:
                continue
            event[key] = [e[key] for e in events]
        event['data'] = np.stack([e['data'] for e in events]).mean(axis=0)

        return event

    def load_event(self, event):
        event = deepcopy(event)
        if self.cached_processing:
            cache = self.f[event['subject']][event['func_session']][self.cached_processing_name]

            if self.feature_selection:
                sorted_indices = self.f_feature_selection[event['subject']][self.feature_selection_key]['sorted_indices']
                top_k = sorted_indices[:, -self.feature_selection_top_k:]
                i, j, k = list(top_k)
                volume_data = cache['data'][event['subject_event_id']]
                event['data'] = volume_data[0, i, j, k]
            else:
                event['data'] = cache['data'][event['subject_event_id']]
            event['affine'] = cache.attrs['affine']

        else:
            run = self.runs[event['run_id']]
            H, W, D, T = run['data'].shape

            t_onset = event['onset'] // 2
            t_start = self.window[0] + t_onset
            t_end = self.window[1] + t_onset
            t = np.arange(max(t_start, 0), min(t_end, T - 1))
            data = torch.from_numpy(run['data'][:, :, :, t])

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
                A[:] = torch.from_numpy(t).float()
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

            in_window = torch.ones_like(torch.from_numpy(t), dtype=bool)
            if t_start < 0:
                pad_length = abs(t_start)
                pre_pad = torch.zeros(H, W, D, pad_length)
                data = torch.cat([pre_pad, data], dim=-1)
                in_window = torch.cat([torch.zeros(pad_length, dtype=bool), in_window])
            if t_end >= T:
                pad_length = abs(t_end - T + 1)
                post_pad = torch.zeros(H, W, D, pad_length)
                data = torch.cat([data, post_pad], dim=-1)
                in_window = torch.cat([in_window, torch.zeros(pad_length, dtype=bool)])

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
            event['in_window'] = in_window

        if self.f_features is not None:
            event['stimulus_id'] = fix_stimulus_id(event['stimulus_id'], list(self.f_features.keys()))
            stimulus_features = self.f_features[event['stimulus_id']]
            feature_keys = self.feature_keys if self.feature_keys is not None else stimulus_features.keys()
            event['features'] = {
                feature_key.replace('.', '_'): torch.tensor(stimulus_features[feature_key][:]).float()
                for feature_key in feature_keys
            }

        return event.copy()


class RawKamitani2019(Dataset):
    def __init__(
            self,
            root: str,
    ):
        self.root = Path(root)
        self.layout = BIDSLayout(root, validate=False, derivatives=True)
        subject_ids = self.layout.get(return_type='id', target='subject')
        subject_ids.sort()

        self.subjects = []
        for subject_id in subject_ids:
            subject = {
                'name': f'sub-{subject_id}',
                'func_sessions': {}
            }
            self.subjects.append(subject)

            anat_query = dict(subject="01", scope="derivatives", extension='nii.gz', session='anatomy',)
            anat_files = [('T1w', {'suffix': 'T1w'}, tio.ScalarImage),
                          ('brain_mask', {'suffix': 'mask'}, tio.LabelMap),
                          ('csf_mask', {'label': 'CSF'}, tio.LabelMap),
                          ('gm_mask', {'label': 'GM'}, tio.LabelMap),
                          ('wm_mask', {'label': 'WM'}, tio.LabelMap)]
            for name, query, constructor in anat_files:
                files = [f for f in self.layout.get(**anat_query, **query) if "space" not in f.entities]
                assert len(files) == 1
                file = files[0]
                subject[name] = constructor(file.path, **file.get_metadata())

            roi_path = self.root / 'sourcedata' / subject['name'] / 'anat'
            for roi_name in ROI_NAMES:
                roi_file_name = f"{subject['name']}_mask_{roi_name}.nii.gz"
                subject[roi_name] = tio.LabelMap(roi_path / roi_file_name)

            session_ids = self.layout.get(return_type='id', subject=subject_id, target='session')
            session_info = [
                ('imagery', 'imagery'),
                ('artificial', 'perceptionArtificialImage'),
                ('letter', 'perceptionLetterImage'),
                ('natural_test', 'perceptionNaturalImageTest'),
                ('natural_training', 'perceptionNaturalImageTraining'),
            ]

            for func_session_type, func_session_name in session_info:
                subject['func_sessions'][func_session_type] = runs = []

                session_ids_of_type = [session_id for session_id in session_ids
                                       if session_id.startswith(func_session_name)]
                bold_files = self.layout.get(subject=subject_id,
                                             suffix='bold',
                                             extension='nii.gz',
                                             session=session_ids_of_type,
                                             space='T1w',
                                             scope='derivatives')

                for bold_file in bold_files:
                    entities = bold_file.get_entities()

                    events_file = self.layout.get(subject=subject_id,
                                                  run=entities['run'],
                                                  session=entities['session'],
                                                  suffix='events')
                    assert len(events_file) == 1
                    events = events_file[0].get_df()

                    confounds_file = self.layout.get(subject=subject_id,
                                                     run=entities['run'],
                                                     session=entities['session'],
                                                     desc='confounds',
                                                     extension='tsv')
                    assert len(confounds_file) == 1
                    confounds = confounds_file[0].get_df()

                    metadata = bold_file.get_metadata()
                    bold_img = tio.ScalarImage(bold_file.path, events=events, confounds=confounds, **metadata)
                    runs.append(bold_img)


class Kamitani2019(Dataset):
    """
    The dataset from the papers:

    End-to-End Deep Image Reconstruction From Human Brain Activity
    https://www.frontiersin.org/articles/10.3389/fncom.2019.00021/full

    Deep image reconstruction from human brain activity
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633
    """
    def __init__(
            self,
            root: str,
            derivatives: bool = False,
    ):
        self.root = Path(root)
        self.subject_names = ['sub-01', 'sub-02', 'sub-03']

        self.subjects = []
        for subject_name in self.subject_names:
            subject_path = self.root / subject_name
            subject_dict = {'name': subject_name, 'path': subject_path}

            subfolder = 'ses-anatomy'
            t1_path = subject_path / subfolder / 'anat' / f'{subject_name}_{subfolder}_T1w.nii.gz'
            subject_dict['t1'] = tio.ScalarImage(t1_path)

            roi_path = self.root / 'sourcedata' / subject_name / 'anat'
            for roi_name in ROI_NAMES:
                roi_file_name = f"{subject_name}_mask_{roi_name}.nii.gz"
                subject_dict[roi_name] = tio.LabelMap(roi_path / roi_file_name)

            subject_dict['sessions'] = sessions = {}
            for info in SESSION_INFO:
                session_type, folder_name, task_type = info
                sessions[session_type] = sessions_of_type = []

                num_sessions = infer_num_sessions(subject_path, folder_name)
                for i in range(num_sessions):
                    session_num = i + 1
                    session_name = f'ses-{folder_name}{session_num:02}'
                    session_path = subject_path / session_name
                    session_dict = {}

                    t2_name = f'{subject_name}_{session_name}_inplaneT2.nii.gz'
                    session_dict['t2'] = tio.ScalarImage(session_path / 'anat' / t2_name)

                    session_dict['runs'] = runs = []
                    num_runs = infer_num_runs(session_path / 'func')
                    for j in range(num_runs):
                        run_num = j + 1
                        run_name_root = f'{subject_name}_{session_name}_task-{task_type}_run-{run_num:02}'
                        bold_path = session_path / 'func' / f'{run_name_root}_bold.nii.gz'
                        bold_json_path = session_path / 'func' / f'{run_name_root}_bold.json'
                        events_path = session_path / 'func' / f'{run_name_root}_events.tsv'

                        with bold_json_path.open() as f:
                            bold_data = json.load(f)
                        events = pd.read_csv(events_path, sep='\t')

                        runs.append(tio.ScalarImage(path=bold_path, channels_last=True, events=events, bold_data=bold_data))

                    sessions_of_type.append(session_dict)

            self.subjects.append(tio.Subject(subject_dict))


if __name__ == '__main__':
    #root = "X:\\Datasets\\Deep-Image-Reconstruction\\derivatives\\kamitani-preprocessed"
    #dataset = Kamitani2019H5Preprocessed(root,
    #                                     subjects=['sub-02',],
    #                                     func_sessions=['natural_training', 'imagery'],
    #                                     features_path='X:\\Datasets\\Deep-Image-Reconstruction\\derivatives\\RN50x16-features.hdf5',)
    #X_key = 'VoxelData'
    #Y_key = 'visual.layer4.7.bn3'
    #data = dataset.get_data(brain_keys=[X_key, 'image_index'], feature_keys=[Y_key])
    path = 'C:\\Datasets\\Deep-Image-Reconstruction\\derivatives'

    subjects = ['sub-01', 'sub-02', 'sub-03']
    sessions = ['imagery']

    target_shape = (72, 88, 74)
    preprocessing_params = dict(
        window=(2, 10),
        window_kernel=[1. / 8.] * 8,
        #transform=tio.CropOrPad(target_shape=target_shape),
        drop_out_of_window_events=True,
        normalization='voxel_linear_trend',
    )

    root = "C:\\Datasets\\Deep-Image-Reconstruction\\"
    h5_path = Path(root) / "derivatives" / "kamitani2019.hdf5"
    dataset = Kamitani2019H5(h5_path, subjects=subjects, func_sessions=sessions, **preprocessing_params)