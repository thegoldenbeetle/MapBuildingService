import lightning as L
from lightning.pytorch.cli import LightningCLI

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from map_builder_service.lanenet import LaneNet
from map_builder_service.dataset import LaneDataModule


def main():
    data_module = LaneDataModule(
        "data/proc/dataset",
        line_radius=30.0,
        batch_size=2,
    )
    model = LaneNet()
    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        patience=3,
        verbose=True,
        mode="max",
    )
    trainer = L.Trainer(
        max_epochs=100, accelerator="auto", devices=1, callbacks=[early_stop_callback],
    )
    trainer.fit(model, data_module)
    return 0


def cli_main():
    cli = LightningCLI(LaneNet, LaneDataModule)

if __name__ == "__main__":
    exit(cli_main())
