import multiprocessing
from datetime import datetime

from tensorboardX import SummaryWriter
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from utils import nju_cpol_dataloader
from utils import interpolator_utils
from utils import *
from tqdm import tqdm
import time
import torch
from loguru import logger

from utils.nju_cpol_dataloader import NjuCpolCoupledDataset
from utils.rain_model import QuantitativeRainModel

epoch_num = 10000
device = 'cuda:0'
batch_size = 64
log_dir_rain = './log/rain'
loader_num_workers = 96

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
loader_num_workers = min(multiprocessing.cpu_count(), loader_num_workers)

def _entry():
    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 创建基于当前时间的目录
    log_dir = os.path.join(log_dir_rain, current_time)
    model_save_dir = os.path.join(log_dir, 'model')

    writer = SummaryWriter(log_dir_rain)

    if not os.path.exists(log_dir_rain):
        os.makedirs(log_dir_rain)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    dataset_train = NjuCpolCoupledDataset(dataset_cpol_dir_train, batch_size=batch_size, use_cache=False, device=device, interpolator=interpolator_utils.nearest_zero)
    dataset_test = NjuCpolCoupledDataset(dataset_cpol_dir_test, batch_size=batch_size, use_cache=False, device=device, interpolator=interpolator_utils.nearest_zero)

    # Train the model to fit the data and solve for the unknown parameters
    model = QuantitativeRainModel(vector_length=65536, device=device)

    # Define the optimizer to use for training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    for epoch in range(epoch_num):
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=loader_num_workers)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=loader_num_workers)

        pbar = tqdm(dataloader_train, desc='Train')
        batch_index = 0
        model.train()
        for batch in pbar:
            optimizer.zero_grad()
            y_flatten_exp = batch[0].reshape(batch_size, -1)
            y_flatten = torch.log10(y_flatten_exp + 1)

            x_flatten = batch[1].reshape(batch_size, 2, -1)
            y_pred = model(x_flatten)

            # loss = model.loss_of(y_flatten, y_pred)
            loss = criterion(y_pred, y_flatten)
            pbar.set_postfix({
                'Epoch': epoch,
                'Loss': loss.item(),
                'Mean-A': model.a.mean().item(),
                'Mean-B': model.b.mean().item(),
                'Mean-C': model.c.mean().item()
            })

            writer.add_scalar('train_loss', loss.item(), epoch * len(dataloader_train) + batch_index)
            writer.add_scalar('train_mean_a', model.a.mean().item(), epoch * len(dataloader_train) + batch_index)
            writer.add_scalar('train_mean_b', model.b.mean().item(), epoch * len(dataloader_train) + batch_index)
            writer.add_scalar('train_mean_c', model.c.mean().item(), epoch * len(dataloader_train) + batch_index)

            loss.backward()
            optimizer.step()
            batch_index += 1
        pbar.close()

        if epoch % 5 == 0:
            model.eval()

            pbar = tqdm(dataloader_test, desc='Test')
            batch_index = 0
            loss = 0
            for batch in pbar:
                y_flatten_exp = batch[0].reshape(batch_size, -1)
                y_flatten = torch.log10(y_flatten_exp + 1)

                x_flatten = batch[1].reshape(batch_size, 2, -1)
                y_pred = model(x_flatten)

                loss = criterion(y_pred, y_flatten)
                pbar.set_postfix({
                    'Epoch': epoch,
                    'Loss': loss.item(),
                    'Mean-A': model.a.mean().item(),
                    'Mean-B': model.b.mean().item(),
                    'Mean-C': model.c.mean().item()
                })

                writer.add_scalar('test_loss', loss.item(), epoch * len(dataloader_train) + batch_index)
                writer.add_scalar('test_mean_a', model.a.mean().item(), epoch * len(dataloader_train) + batch_index)
                writer.add_scalar('test_mean_b', model.b.mean().item(), epoch * len(dataloader_train) + batch_index)
                writer.add_scalar('test_mean_c', model.c.mean().item(), epoch * len(dataloader_train) + batch_index)
                batch_index += 1
            pbar.close()

            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_save_path = os.path.join(model_save_dir, f'model_{epoch}_{current_time}_{loss.item()}.pth')
            torch.save(model.state_dict(), model_save_path)


    # Print the optimized parameters
    print("Optimized parameters:", model.a, model.b, model.c)


if __name__ == '__main__':
    _entry()
