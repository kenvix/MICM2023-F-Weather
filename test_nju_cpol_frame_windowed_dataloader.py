from utils import nju_cpol_dataloader
from utils import *
from tqdm import tqdm


if __name__ == '__main__':
    dataloader_train = nju_cpol_dataloader.NjuCpolFrameWindowedDataset.dataloader(dataset_cpol_dir_train, window_size=(2, 10), batch_size=4, norm=True)
    dataloader_test = nju_cpol_dataloader.NjuCpolFrameWindowedDataset.dataloader(dataset_cpol_dir_test, window_size=(2, 10), batch_size=4, norm=True)
    pbar = tqdm(dataloader_test)
    for batch in pbar:
        pbar.set_description(f"TEST {batch.shape}")

    pbar = tqdm(dataloader_train)
    for batch in pbar:
        pbar.set_description(f"TRAIN {batch.shape}")
