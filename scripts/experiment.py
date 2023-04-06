import sys
from dvclive.lightning import DVCLiveLogger
from lightning.pytorch.cli import LightningCLI

from map_builder_service.dataset import LaneDataModule
from map_builder_service.lanenet import LaneNet
from map_builder_service.segmodel import SegModel


def cli_main():
    LightningCLI(
        datamodule_class=LaneDataModule, save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    sys.exit(cli_main())
