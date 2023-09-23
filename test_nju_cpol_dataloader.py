from utils import nju_cpol_dataloader
from utils import config, dataset_cpol_dir, dataset_base_dir
from tqdm import tqdm


if __name__ == '__main__':
    dataloader = nju_cpol_dataloader.NjuCpolBaseDataset.dataloader(f'{dataset_cpol_dir}', window_size=10, window_step=1, padding=True)
    pbar = tqdm(dataloader)
    for batch in pbar:
        print(batch.shape)
