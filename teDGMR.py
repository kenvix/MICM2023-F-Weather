import torch
from utils import nju_cpol_dataloader
from dgmr import DGMR
from tqdm import tqdm
import torch.nn as nn
def train1():
    ataset_cpol_dir_train = r'./datasets/NJU_CPOL_update2308'
    dataset_cpol_dir_test = r'F:\LX\Datasets\NJU_CPOL_update2308-vvvv'

    ckpt_path = r"\\192.168.0.97\数学建模\GDMR\1.km\epoch=1-step=10500.ckpt"

    batch_size = 2
    channels = 1

    dataloader_test = nju_cpol_dataloader.NjuCpolFrameWindowedDataset.dataloader(dataset_cpol_dir_test,
                                                                                 window_size=(2, 4),
                                                                                 sub_dir='1.0km',
                                                                                 dim_list=['dBZ'],
                                                                                 batch_size=batch_size, norm=False)

    model = DGMR(input_channels=1, forecast_steps=4, output_shape=256, latent_channels=384 * 2,
                 context_channels=384, num_samples=6)

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    pbar = tqdm(dataloader_test)

    hidden_state = None
    model.train()

    for i, batch in enumerate(pbar):
        batch_size = batch.shape[0]
        windowsize = 4
        input_channels = 1
        data = torch.as_tensor(batch, dtype=torch.float32).resize(2, batch_size, windowsize, 1, 256, 256)
        images, next_images = data[0, :, :, 0, :, :], data[1, :, :, 0, :, :]

        images = images.resize(batch_size, windowsize, input_channels, 256, 256)

        # print this
        next_images = next_images.resize(batch_size, windowsize, input_channels, 256, 256)

        output = model(images, hidden_state)

        # print this
        pre = output[0][0].resize(batch_size, 10, 256, 256)




if __name__ == '__main__':
    from fire import Fire
    Fire(train1)
