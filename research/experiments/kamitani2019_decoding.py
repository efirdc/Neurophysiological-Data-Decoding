from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as T
import torch.nn.functional as F
import fire
from torchmetrics.functional import r2_score

from ..data.kamitani_2019 import Kamitani2019H5
from ..models.fmri_decoders import ConvolutionalDecoder
from ..metrics.metrics import best_r2_score
from ..utils.utils import to_device


class DecodingExperiment:
    def __init__(
            self,
            device: torch.device,
            training_dataset: Kamitani2019H5,
            validation_dataset: Kamitani2019H5,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
    ):
        self.device = device
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(
            self,
            max_iterations: int,
            training_batch_size: int,
            validation_batch_size: int,
    ):

        training_dataloader = DataLoader(self.training_dataset, shuffle=True, batch_size=training_batch_size,)
        validation_dataloader = DataLoader(self.validation_dataset, shuffle=True, batch_size=validation_batch_size,)

        def get_data_iterator(loader):
            while True:
                for batch in loader:
                    yield batch

        training_data_iterator = get_data_iterator(training_dataloader)
        validation_data_iterator = get_data_iterator(validation_dataloader)
        # validation_batch = next(validation_data_iterator)
        # validation_batch = to_device(validation_batch, self.device)

        for _ in range(max_iterations):
            batch = next(training_data_iterator)
            batch = to_device(batch, self.device)

            X = batch['data']
            y = batch['features']

            self.model.train()
            y_pred = self.model(X)
            losses = {feature_name: self.criterion(y_pred[feature_name], y[feature_name])
                      for feature_name in y_pred.keys()}
            loss = sum(losses.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            with torch.no_grad():
                train_r2 = {feature_name: r2_score(y_pred[feature_name].flatten(start_dim=1),
                                                   y[feature_name].flatten(start_dim=1))
                            for feature_name in y_pred.keys()}
            print("train_r2")
            for feature_name, r2 in train_r2.items():
                print(f'    {feature_name}: {r2:.3f}')
            '''
            with torch.no_grad():
                X = validation_batch['eeg_data']
                y_val_pred = self.model(X)
                y_val = validation_batch['latent']
                val_r2 = best_r2_score(y_val_pred, y_val).item()'''



def main(
        kamitani_2019_root: str,
        features_file_name: str,
        device: str = 'cuda',
):

    derivatives_path = Path(kamitani_2019_root) / 'derivatives'
    device = torch.device(device)
    dataset_params = {
        'h5_path': str(derivatives_path / 'kamitani2019.hdf5'),
        'subjects': ['sub-02'],
        'func_sessions': ['natural_training'],
        'window': (0, 7),
        'drop_out_of_window_events': False,
        'split': 'train',
        'features_path': str(derivatives_path / features_file_name),
    }
    training_dataset = Kamitani2019H5(**dataset_params, folds=(0, 1, 2, 3))
    validation_dataset = Kamitani2019H5(**dataset_params, folds=(4,))

    criterion = nn.MSELoss()

    model = ConvolutionalDecoder(in_channels=7,
                                 extractor_channels=(40, 80, 120),
                                 decoder_channels=(512, 256, 256, 128, 128),
                                 decoder_base_shape=(512, 6, 6),
                                 decoder_output_shapes=training_dataset.feature_shapes,)
    model.eval()
    model.to(device)

    optimizer = Adam(params=model.parameters(), lr=1e-4)
    experiment = DecodingExperiment(device, training_dataset, validation_dataset, model, criterion, optimizer)

    experiment.train(max_iterations=10000, training_batch_size=4, validation_batch_size=4)


if __name__ == "__main__":
    fire.Fire({
        "experimentv1": main
    })
