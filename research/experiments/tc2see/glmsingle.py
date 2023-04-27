import os
import sys
from pprint import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
from tqdm.notebook import tqdm
import nibabel as nib
import glmsingle
from glmsingle.glmsingle import GLM_single
from scipy.ndimage import zoom

from research.experiments.tc2see.analysis import (
    get_design_matrix,
)
from research.experiments.tc2see.data_loading import (
    Data,
)


if __name__ == "__main__":
    subject = "01"  # subjects 01 and 02
    task = "bird"
    space = "T1w"
    #space = "MNI152NLin2009cAsym"
    tr = 1.97
    version = 1
    dataset_path = Path(f"D:\\Datasets\\TC2See_v{version}")
    num_runs = 6 if version == 1 else 8

    # Load runs
    resample_tr = 0.1
    fmri_mask = None
    fmri_batch = []
    events_batch = []
    for run in tqdm(range(1, num_runs + 1)):
        data = Data(str(dataset_path), subject, task, run, space)
        events_batch.append(data.events)

        fmri_data = data.fmri_img.get_fdata()
        if fmri_mask is None:
            fmri_mask = data.mask.get_fdata().astype(bool)
        fmri_data_masked = fmri_data[fmri_mask]
        fmri_data_masked = zoom(fmri_data_masked, zoom=(1, tr / resample_tr))
        fmri_batch.append(fmri_data_masked)

    # Collect all conditions
    conditions = []
    for run_id, events in enumerate(events_batch):
        for id, event in events.iterrows():
            if not event['stimulus'].endswith('png'):
                continue
            condition_name = Path(event['stimulus']).stem.split('.')[1]
            conditions.append(condition_name)
    conditions = list(set(conditions))
    conditions.sort()
    conditions = {condition: i  for i, condition in enumerate(conditions)}
    C = len(conditions)

    _, T = fmri_batch[0].shape

    design_batch = []
    for run_id, events in enumerate(events_batch):
        _, T = fmri_batch[run_id].shape
        design_matrix = np.zeros(shape=(T, C))
        for id, event in events.iterrows():
            if not event['stimulus'].endswith('png'):
                continue
            condition_name = Path(event['stimulus']).stem.split('.')[1]
            c = conditions[condition_name]
            t = round(event.tr / resample_tr)
            design_matrix[t, c] = 1
        design_batch.append(design_matrix)

    # Add missing conditions to all design matrices, and sort them to be in the same order
    all_conditions = []
    for design_matrix in design_batch:
        all_conditions += list(design_matrix.columns)
    all_conditions = list(set(all_conditions))
    all_conditions.sort()

    print(all_conditions)
    print(len(all_conditions))

    for i, design_matrix in enumerate(design_batch):
        for condition in all_conditions:
            if condition not in design_matrix:
                design_matrix[condition] = 0
        design_batch[i] = design_matrix.reindex(sorted(design_matrix.columns), axis=1)

    glmsingle_obj = GLM_single(dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[1,1,1,1],
        wantmemoryoutputs=[1,1,1,1],
    ))

    pprint(glmsingle_obj.params)

    output_path = dataset_path / 'derivatives_TC2See_prdgm/glmsingle'
    results_glmsingle = glmsingle_obj.fit(
        design=design_batch,
        data=fmri_batch,
        stimdur=2,
        tr=data.tr,
        outputdir=str(output_path),
    )