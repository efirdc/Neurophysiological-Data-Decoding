from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import warnings
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import pandas as pd
import numpy as np
import mne

from .things_dataset import ThingsDataset


class ThingsEEG(Dataset):
    SEQUENCE_LENGTH = 309
    TEST_SEQUENCE_LENGTH = 200

    def __init__(
            self,
            things_dataset: ThingsDataset,
            things_eeg_path: str,
            source: str = "brainvision",
            window: Tuple[float, float] = (0.0, 0.2),
            stimulus_window: Optional[Tuple[int, int]] = None,
            include_participants: Optional[Sequence[str]] = None,
            exclude_participants: Optional[Sequence[str]] = None,
            folds: Sequence[Union[str, int]] = (0, 1, 2, 3, 4),
            include_test_block: bool = False,
            verbose: bool = False,
    ):
        if things_dataset.supplementary_path is None:
            raise RuntimeError("Things-supplementary data is required to load this dataset.")

        self._preloaded = False
        self.preloaded_data = {}

        self.things_dataset = things_dataset
        self.things_eeg_path = Path(things_eeg_path)
        self.stimulus_window = stimulus_window
        self.participant_data = pd.read_csv(self.things_eeg_path / "data_general/data/participants.tsv", sep='\t')
        with (self.things_dataset.supplementary_path / "eeg-split.json").open() as f:
            self.train_test_split = json.load(f)

        self.events = []
        self.participants = {}
        for participant in self.participant_data.to_dict('index').values():
            if participant['exclude'] == 1:
                continue

            participant_id = participant['participant_id']

            if include_participants and participant_id not in include_participants:
                continue
            if exclude_participants and participant_id in exclude_participants:
                continue

            participant_path = Path(things_eeg_path) / participant_id
            brainvision_path = participant_path / "data" / participant_id / "eeg"
            eeglab_path = participant_path / "data" / "derivatives" / "eeglab"
            participant['paths'] = {
                p.suffix[1:]: p
                for p in [*brainvision_path.iterdir(), *eeglab_path.iterdir()]
            }
            if source == "brainvision":
                participant['raw'] = mne.io.read_raw_brainvision(participant['paths']['vhdr'], verbose=verbose)
                event_id = {'Event/E  1': 1}
            elif source == "eeglab":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    participant['raw'] = mne.io.read_raw_eeglab(participant['paths']['set'], verbose=verbose)
                event_id = {"E  1": 1}
            else:
                raise ValueError(f"Invalid data source: \"{source}\"")

            participant['rsvp_events'] = rsvp_events = pd.read_csv(participant['paths']['tsv'], sep='\t')
            image_names = [stimname.split(".")[0] for stimname in rsvp_events['stimname']]
            rsvp_events['image_name'] = image_names
            rsvp_events.set_index('image_name')

            raw = participant['raw']
            raw.load_data()
            filter_freqs = [50, 100, 150, 200, 250, 300, 350]
            raw.notch_filter(filter_freqs, filter_length='auto', phase='zero', verbose=verbose)
            raw = raw.filter(l_freq=0.1, h_freq=100.)
            # raw = raw.set_eeg_reference(ref_channels='average')
            participant['raw'] = raw

            events, event_id = mne.events_from_annotations(participant['raw'],
                                                           event_id=event_id,
                                                           verbose=verbose)
            epochs = mne.Epochs(participant['raw'],
                                events,
                                picks=list(range(63)),
                                tmin=window[0],
                                tmax=window[1] - (1 / participant['raw'].info['sfreq']),
                                baseline=window,
                                verbose=verbose,
                                preload=False,
                                decim=1)
            participant['epochs'] = epochs

            num_events = len(rsvp_events)

            include_block_sequences = []
            for fold in folds:
                include_block_sequences += self.train_test_split[participant_id][str(fold)]
            if include_test_block:
                include_block_sequences.append(-1)
            include_block_sequences = np.array(include_block_sequences)

            block_sequences = np.array(participant['rsvp_events']['blocksequencenumber'])
            mask = np.any(block_sequences[:, None] == include_block_sequences[None, :], axis=1)

            event_ids = np.arange(num_events)[mask]

            self.participants[participant_id] = participant
            self.events += [(participant_id, event_index) for event_index in event_ids]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        if self._preloaded:
            return {k: v[idx] for k, v in self.preloaded_data.items()}
        else:
            participant_id, event_index = self.events[idx]
            participant = self.participants[participant_id]
            event_info = participant['rsvp_events'].loc[event_index]
            epochs = participant['epochs']
            epoch = epochs[event_index]
            eeg_data = torch.from_numpy(epoch.get_data()[0]).float()

            if self.stimulus_window is None:
                image_name = event_info['image_name']
                things_image = self.things_dataset.images_map[image_name]
                latent = torch.from_numpy(things_image['latents']['z_mean']).float()
            else:
                is_test_stim = event_info['isteststim'] == 1
                sequence_length = ThingsEEG.TEST_SEQUENCE_LENGTH if is_test_stim else ThingsEEG.SEQUENCE_LENGTH
                sequence_index = event_info['presentationnumber']
                window_start = np.clip(self.stimulus_window[0], -sequence_index, sequence_length - sequence_index)
                window_end = np.clip(self.stimulus_window[1], -sequence_index, sequence_length - sequence_index)
                start_event_index = event_index + window_start
                end_event_index = event_index + window_end

                event_infos = participant['rsvp_events'].loc[start_event_index:(end_event_index - 1)]
                image_name = event_infos['image_name']
                things_image = [self.things_dataset.images_map[name] for name in image_name]
                latent = torch.stack([
                    torch.from_numpy(image['latents']['z_mean']).float()
                    for image in things_image
                ])

            return {"eeg_data": eeg_data, "latent": latent, "things_image": things_image}

    def preload(self):
        for participant in self.participants.values():
            participant['epochs'].load_data()
        self.preloaded_data = self.collate([event for event in self])
        self._preloaded = True
        return self.preloaded_data

    def collate(self, batch):
        out = {}
        out['eeg_data'] = torch.stack([event['eeg_data'] for event in batch])
        if self.stimulus_window is None:
            out['latent'] = torch.stack([event['latent'] for event in batch])
        else:
            out['latent'] = [event['latent'] for event in batch]
        out['things_image'] = [event['things_image'] for event in batch]
        return out
