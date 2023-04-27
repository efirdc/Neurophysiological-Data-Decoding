import sys
import os
from typing import Dict, Any, Sequence

import numpy as np
import torch
#from numpy.typing import ArrayLike
import nibabel.freesurfer.io as fio

def read_patch(fname):
    """
    loads a FreeSurfer binary patch file
    This is a Python adaptation of Bruce Fischl's read_patch.m (FreeSurfer Matlab interface)
    """
    def read_an_int(fid):
        return np.fromfile(fid, dtype='>i4', count=1).item()

    patch = {}
    with open(fname, 'r') as fid:
        ver = read_an_int(fid) # '> signifies big endian'
        if ver != -1:
            raise Exception('incorrect version # %d (not -1) found in file'.format(ver))

        patch['npts'] = read_an_int(fid)

        rectype = np.dtype([('ind', '>i4'), ('x', '>f'), ('y', '>f'), ('z','>f')])
        recs = np.fromfile(fid, dtype=rectype, count=patch['npts'])

        recs['ind'] = np.abs(recs['ind'])-1 # strange correction to indexing, following the Matlab source...
        patch['vno'] = recs['ind']
        patch['x'] = recs['x']
        patch['y'] = recs['y']
        patch['z'] = recs['z']

        # make sure it's sorted
        index_array = np.argsort(patch['vno'])
        for field in ['vno', 'x', 'y', 'z']:
            patch[field] = patch[field][index_array]

    return patch


def is_sequence(x):
    if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        return True
    if isinstance(x, Sequence) and not isinstance(x, str):
        return True
    return False


def index_unsorted(x, indices):
    """
    Returns x[indices].
    Use if x is an array type where indices must be strictly increasing (i.e. an h5py.Dataset).
    This will sort indices and then remove duplicates before indexing,
    The resulted array is then "unsorted" and "repeated" to respect the original indices.
    """
    unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
    return x[unique_indices][inverse_indices]


def merge_dicts(source: Dict, dest: Dict):
    for k, v in source.items():
        if isinstance(v, Dict):
            if k not in dest:
                dest[k] = {}
            merge_dicts(source[k], dest[k])
        else:
            if k in dest:
                raise ValueError()
            else:
                dest[k] = v


def nested_insert(d: Dict, keys: Sequence[Any], value: Any):
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        assert isinstance(d[k], Dict)
        d = d[k]

    final_key = keys[-1]
    if final_key not in d:
        d[final_key] = value

    elif isinstance(d[final_key], Dict):
        assert isinstance(value, Dict)
        merge_dicts(value, d[final_key])

    else:
        d[final_key] = value


def nested_select(d: Dict, keys: Sequence[Any]):
    key = keys[0]
    if key is None:
        key = list(d.keys())
    if is_sequence(key):
        return [nested_select(d[k], keys[1:]) for k in key]
    elif len(keys) > 1:
        return nested_select(d[key], keys[1:])
    else:
        return d[key]


def get_data_iterator(loader, new_epoch_callback=None):
    while True:
        for batch in loader:
            yield batch
        if new_epoch_callback is not None:
            new_epoch_callback()


def product(seq: Sequence):
    out = 1
    for elem in seq:
        out *= elem
    return out


class DisablePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
