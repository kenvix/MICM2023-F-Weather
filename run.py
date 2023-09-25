from pytorch_lightning import (
    LightningDataModule,
)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dgmr import DGMR
from utils import nju_cpol_dataloader

NUM_INPUT_FRAMES = 200
NUM_TARGET_FRAMES = 10

def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES - NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames

class DGMRDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        num_workers: int = 1,
        pin_memory: bool = True,
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        """
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )
        self.batch_size = 2

    def train_dataloader(self):
        dataset_cpol_dir_train = r'./datasets/NJU_CPOL_update2308'
        dataloader_train = nju_cpol_dataloader.NjuCpolFrameWindowedDataset.dataloader(dataset_cpol_dir_train,
                                                                                      window_size=(2, 4),
                                                                                      batch_size=self.batch_size, norm=True)
        return dataloader_train

    def val_dataloader(self):
        dataset_cpol_dir_test = r'./datasets/NJU_CPOL_update2308_test'

        dataloader_test = nju_cpol_dataloader.NjuCpolFrameWindowedDataset.dataloader(dataset_cpol_dir_test,
                                                                                     window_size=(2, 4),
                                                                                     batch_size=self.batch_size, norm=True)
        return dataloader_test

model_checkpoint = ModelCheckpoint(
    monitor="train/g_loss",
    dirpath="./dgmr_checkpoint/",
    filename="best",
)

trainer = Trainer(
    max_epochs=10,
    callbacks=[model_checkpoint],
    precision=32,
    resume_from_checkpoint="\\\\192.168.0.97\\数学建模\\GDMR\\1.km\\epoch=1-step=10500.ckpt",
    accelerator="auto", devices=1
)
model = DGMR(input_channels=1,forecast_steps=4,output_shape=256,latent_channels = 384*2,
        context_channels = 384,num_samples=6)

def _entry(mode='test'):
    if mode == 'test':
        model.eval()
        datamodule = DGMRDataModule()

        batch_size = batch.shape[0]
        windowsize = 4
        input_channels = 1
        data = torch.as_tensor(batch, dtype=torch.float32).resize(2, batch_size, windowsize, 1, 256, 256)
        images, future_images = data[0, :, :,0, :, :], data[1, :, :,0, :, :]
        images = images.resize(batch_size,windowsize,input_channels,256,256)
        future_images =  future_images.resize(batch_size,windowsize,input_channels,256,256)
    else:
        model.train()
        datamodule = DGMRDataModule()
        trainer.fit(model, datamodule)
