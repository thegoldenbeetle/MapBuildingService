import lightning as L
from lightning.pytorch.cli import LightningCLI

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from map_builder_service.lanenet import LaneNet
from map_builder_service.dataset import LaneDataModule


def cli_main():
    cli = LightningCLI(LaneNet, LaneDataModule)

if __name__ == "__main__":
    exit(cli_main())
