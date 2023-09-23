from utils import nju_cpol_dataloader
from utils import interpolator_utils
from utils import *
from tqdm import tqdm


if __name__ == '__main__':
    dataloader_train = nju_cpol_dataloader.NjuCpolCoupledDataset.dataloader(dataset_cpol_dir_train, interpolator=interpolator_utils.nearest_zero)
    dataloader_test = nju_cpol_dataloader.NjuCpolCoupledDataset.dataloader(dataset_cpol_dir_test, interpolator=interpolator_utils.nearest_zero)
    pbar = tqdm(dataloader_test)
    for batch in pbar:
        pbar.set_description(f"TEST x: {batch[0].shape} y: {batch[1].shape}")

    pbar = tqdm(dataloader_train)
    for batch in pbar:
        pbar.set_description(f"TRAIN x: {batch[0].shape} y: {batch[1].shape}")
