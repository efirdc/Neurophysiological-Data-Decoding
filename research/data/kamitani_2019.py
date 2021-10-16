from pathlib import Path
from typing import Callable, Optional
from copy import deepcopy
import json

import numpy as np
from torch.utils.data import Dataset
import torchio as tio
import pandas as pd
from PIL import Image


def infer_num_sessions(subject_path: Path, folder_name: str) -> int:
    return len([p.name for p in subject_path.iterdir() if folder_name in p.name])


def infer_num_runs(session_path: Path) -> int:
    file_names = [p.name for p in session_path.iterdir()]
    run_ids = [int(file_name.split('_')[3][4:6]) for file_name in file_names]
    return max(run_ids)


class Kamitani2019(Dataset):
    """
    The dataset from the papers:

    End-to-End Deep Image Reconstruction From Human Brain Activity
    https://www.frontiersin.org/articles/10.3389/fncom.2019.00021/full

    Deep image reconstruction from human brain activity
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633
    """

    # session_type, folder_name, num_sessions, num_runs, task_type
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

    def __init__(
            self,
            root: str,
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
            for roi_name in self.ROI_NAMES:
                roi_file_name = f"{subject_name}_mask_{roi_name}.nii.gz"
                subject_dict[roi_name] = tio.LabelMap(roi_path / roi_file_name)

            subject_dict['sessions'] = sessions = {}
            for info in self.SESSION_INFO:
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
