from utils import nju_cpol_dataloader
from utils import config, dataset_cpol_dir, dataset_base_dir
from tqdm import tqdm


if __name__ == '__main__':
    dataloader = nju_cpol_dataloader.NjuCpolCoupledDataset.dataloader(f'{dataset_cpol_dir}')
    pbar = tqdm(dataloader)
    for batch in pbar:
        print(batch.shape)