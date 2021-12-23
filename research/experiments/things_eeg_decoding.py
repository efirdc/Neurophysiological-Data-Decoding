import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as T
import torch.nn.functional as F
import fire
import torchmetrics

from ..data import *
from ..models import *
from ..metrics.loss_functions import SoftMinMSE
from ..metrics.metrics import best_r2_score
from ..utils.utils import to_device


class DecodingExperiment:
    def __init__(
            self,
            device: torch.device,
            training_dataset: ThingsEEG,
            validation_dataset: ThingsEEG,
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

        training_dataloader = DataLoader(self.training_dataset, shuffle=True, batch_size=training_batch_size,
                                         collate_fn=self.training_dataset.collate)
        validation_dataloader = DataLoader(self.validation_dataset, shuffle=True, batch_size=validation_batch_size,
                                           collate_fn=self.validation_dataset.collate)

        def get_data_iterator(loader):
            while True:
                for batch in loader:
                    yield batch

        training_data_iterator = get_data_iterator(training_dataloader)
        validation_data_iterator = get_data_iterator(validation_dataloader)
        validation_batch = next(validation_data_iterator)
        validation_batch = to_device(validation_batch, self.device)

        for _ in range(max_iterations):
            batch = next(training_data_iterator)
            batch = to_device(batch, self.device)

            X = batch['eeg_data']
            y = batch['latent']

            self.model.train()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            train_r2 = best_r2_score(y_pred, y).item()
            with torch.no_grad():
                X = validation_batch['eeg_data']
                y_val_pred = self.model(X)
                y_val = validation_batch['latent']
                val_r2 = best_r2_score(y_val_pred, y_val).item()
            print(f"train_r2={train_r2:.3f}, val_r2={val_r2:.3f}")


def main(
        things_images_path: str,
        things_supplementary_path: str,
        things_eeg_path: str,
        device: str = 'cuda',
):
    device = torch.device(device)
    transform = T.Compose([
        T.Resize(256),
        T.ToTensor(),
    ])

    things_dataset = ThingsDataset(
        root=things_images_path,
        transform=transform,
        supplementary_path=things_supplementary_path,
        latent_name="bigbigan-resnet50",
    )

    window_width = 256 / 1000
    things_eeg_params = {"things_dataset": things_dataset,
                         "things_eeg_path": things_eeg_path,
                         "include_participants": ['sub-38'],
                         "window": (-window_width, window_width),
                         "stimulus_window": (-1, 2),
                         'verbose': False}
    training_dataset = ThingsEEG(**things_eeg_params, folds=(1, 2, 3, 4))
    validation_dataset = ThingsEEG(**things_eeg_params, folds=(0,))

    criterion = SoftMinMSE()

    model = ConvolutionalDecoder(
        in_samples=256,
        in_channels=63,
        block_channels=[200, 200],
        fully_connected_features=[4096, 120],
        block_params={"num_convs": 1}
    )
    #model = MLPDecoder(features=[512 * 63, 1024, 120])
    model.eval()
    model.to(device)

    optimizer = Adam(params=model.parameters(), lr=1e-4)
    experiment = DecodingExperiment(device, training_dataset, validation_dataset, model, criterion, optimizer)

    experiment.train(max_iterations=10000, training_batch_size=256, validation_batch_size=256)


if __name__ == "__main__":
    fire.Fire({
        "experimentv1": main
    })
