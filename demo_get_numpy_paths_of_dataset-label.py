from utils import nju_cpol_pathloader
from utils import config, dataset_cpol_dir, dataset_base_dir
from tqdm import tqdm


if __name__ == '__main__':
    dataloader = nju_cpol_pathloader.get_numpy_paths_of_dataset_label(f'{dataset_cpol_dir}', dim='kdp-rain')
    pbar = tqdm(dataloader)
    for batch in pbar:
        print(batch.shape)
