import sys

from lightning.pytorch.cli import LightningCLI

from map_builder_service.dataset import LaneDataModule
from map_builder_service.lanenet import LaneNet


def cli_main():
    LightningCLI(LaneNet, LaneDataModule)


if __name__ == "__main__":
    sys.exit(cli_main())
