import time
from typing import Dict, Any, Tuple, Sequence, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import h5py

from research.data.natural_scenes import NSDFold
from research.metrics.metrics import (
    r2_score,
    pearsonr,
    cosine_distance,
    mean_squared_distance,
    contrastive_score,
    two_versus_two,
    evaluate_decoding
)
from pipeline.utils import merge_dicts, nested_insert, get_data_iterator


class NSDExperiment:
    def __init__(
            self,
            train_dataset: Dataset,
            val_dataset: Dataset,
            device: torch.device,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            batch_size: int,
            evaluation_interval: int,
            evaluation_subset_size: Optional[int] = None,
            evaluation_subset_seed: int = 0,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.batch_size = batch_size
        self.evaluation_interval = evaluation_interval
        self.evaluation_subset_size = evaluation_subset_size
        self.evaluation_subset_seed = evaluation_subset_seed

        dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)

        self.data_iterator = get_data_iterator(dataloader)
        self.iteration = 0

    def train_model(self, max_iterations: int, logger: Optional[Callable] = None):
        for i in tqdm(range(max_iterations)):
            evaluation_dict = {}
            if self.iteration % self.evaluation_interval == 0:
                evaluation_dict = self.evaluate(subset_size=self.evaluation_subset_size,
                                                subset_seed=self.evaluation_subset_seed)
            batch = next(self.data_iterator)
            y = batch['stimulus'][0].to(self.device)
            x = batch['betas'][0].to(self.device)

            self.model.train()
            model_out = self.model(x)
            if isinstance(model_out, Tuple):
                y_pred, mu, log_var = model_out
                loss, loss_dict = self.criterion(y, y_pred, mu, log_var)
            else:
                y_pred = model_out
                loss = self.criterion(y, y_pred)
                loss_dict = {'loss': loss}

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            log_dict = {**loss_dict, **evaluation_dict}
            logger(log_dict)

            self.iteration += 1

    def run_all(self, dataset: Dataset):
        Y_pred = []
        Y = []
        stimulus_ids = []
        for elem in dataset:
            x = elem['stimulus']['data']
            stimulus_id = elem['stimulus']['id']
            stimulus_ids.append(stimulus_id)

            y_pred = self.model(x.to(self.device)[None])
            if isinstance(y_pred, Tuple):
                y_pred = y_pred[0]
            Y_pred.append(y_pred.cpu())

            y = elem['betas'][0]
            Y.append(y)

        return torch.stack(Y), torch.cat(Y_pred), torch.tensor(stimulus_ids)

    def get_evaluation_subset(
            self,
            dataset: Dataset,
            subset_size: int,
            subset_seed: int
    ):
        N = len(dataset)
        np.random.seed(subset_seed)
        ids = np.random.choice(N, size=subset_size, replace=False)
        ids.sort()
        dataset = Subset(dataset, ids)
        return dataset

    def evaluate(
        self,
        evaluation_metrics: Sequence[Callable] = (r2_score, pearsonr),
        distance_metrics: Sequence[Callable] = (cosine_distance, mean_squared_distance),
        distance_classification_measures: Sequence[Callable] = (two_versus_two),
        subset_size: Optional[int] = None,
        subset_seed: int = 0,
    ):
        evaluation_dict = {}
        folds = [
            ('train', self.train_dataset),
            ('val', self.val_dataset),
        ]
        if subset_size:
            folds = [
                (fold_name, self.get_evaluation_subset(dataset, subset_size, subset_seed))
                for fold_name, dataset in folds
            ]

        self.model.eval()
        for fold_name, dataset in folds:
            with torch.no_grad():
                Y, Y_pred, _ = self.run_all(dataset)
            evaluation_measures = (evaluation_metrics, distance_metrics, distance_classification_measures)
            merge_dicts(
                source=evaluate_decoding(Y, Y_pred, fold_name, *evaluation_measures),
                dest=evaluation_dict
            )
        return evaluation_dict

    def save(
            self,
            save_file_path: str,
            attributes: Dict[str, Any],
            key: Sequence[str],
    ):
        folds = [
            ('test', self.test, self.test.X, self.test.Y, self.test.Y_ids),
            ('val', self.val, self.val.X, self.val.Y, self.val.Y_ids),
            ('test_averaged', self.test, self.test.X_averaged, self.test.Y_averaged, self.test.Y_averaged_ids),
            ('val_averaged', self.val, self.val.X_averaged, self.val.Y_averaged, self.val.Y_averaged_ids)
        ]
        with h5py.File(save_file_path, 'a') as f:
            key = '/'.join(key)
            group = f.require_group(key)
            for k, v in attributes.items():
                group.attrs[k] = v
            group.attrs['iteration'] = self.iteration
            for fold_name, fold, X, Y, stimulus_ids in folds:
                fold_group = group.require_group(fold_name)
                with torch.no_grad():
                    Y_pred = self.run_all(X)
                fold_group['Y_pred'] = fold.inverse_transform(Y_pred)
                fold_group['stimulus_ids'] = stimulus_ids
