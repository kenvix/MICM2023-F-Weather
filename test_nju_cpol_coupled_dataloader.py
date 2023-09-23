from utils import nju_cpol_dataloader
from utils import *
from tqdm import tqdm


if __name__ == '__main__':
    dataloader_train = nju_cpol_dataloader.NjuCpolCoupledDataset.dataloader(dataset_cpol_dir_train)
    dataloader_test = nju_cpol_dataloader.NjuCpolCoupledDataset.dataloader(dataset_cpol_dir_test)
    pbar = tqdm(dataloader_train)
    for batch in pbar:
        pbar.set_description(f"x: {batch[0].shape} y: {batch[1].shape}")
