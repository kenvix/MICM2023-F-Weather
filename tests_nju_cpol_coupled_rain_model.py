import multiprocessing
from datetime import datetime

from matplotlib import pyplot as plt
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


def visualize_array(data_groundtruth, save=None, file=None):
    # Load the .npy file
    plt.imshow(data_groundtruth, cmap='Blues')
    plt.title(file)
    # plt.gca().invert_yaxis()
    plt.colorbar()
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)
        plt.savefig(f'{save}/{file}.svg')
    else:
        plt.show()
    plt.close()


def _entry(epoch_num=10000, device='cpu', batch_size=64, lr=0.01, log_dir_rain='./log/rain', use_cache=True,
           loader_num_workers=96, sub_dir="1.0km", pretrained=None, visualize=False, test_only=False):
    # try:
    #     torch.multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    loader_num_workers = min(multiprocessing.cpu_count(), loader_num_workers)

    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 创建基于当前时间的目录
    log_dir = os.path.join(log_dir_rain, current_time)
    model_save_dir = os.path.join(log_dir, 'model')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fo = open(f"{log_dir}/info-lr_{lr}-sub_{sub_dir}", "w")
    fo.write("1")
    fo.close()

    writer = SummaryWriter(log_dir)
    
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    dataset_train = NjuCpolCoupledDataset(dataset_cpol_dir_train, batch_size=batch_size, use_cache=use_cache,
                                          device=device,
                                          sub_dir=sub_dir,
                                          interpolator=interpolator_utils.nearest_zero)
    dataset_test = NjuCpolCoupledDataset(dataset_cpol_dir_test, batch_size=batch_size, use_cache=use_cache,
                                         device=device,
                                         sub_dir=sub_dir,
                                         interpolator=interpolator_utils.nearest_zero)

    # Train the model to fit the data and solve for the unknown parameters
    model = QuantitativeRainModel(vector_length=65536, device=device)

    # Define the optimizer to use for training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    if not test_only:
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                      num_workers=loader_num_workers)

    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=loader_num_workers)

    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
        logger.info("Loaded pretrained model from {}".format(pretrained))

    for epoch in range(epoch_num):
        if not test_only:
            pbar = tqdm(dataloader_train, desc='Train')
            batch_index = 0
            model.train()
            for batch in pbar:
                current_batch_size = batch[0].shape[0]
                optimizer.zero_grad()
                y_flatten_exp = batch[0].reshape(current_batch_size, -1)
                y_flatten = torch.log10(y_flatten_exp + 1)

                x_flatten = batch[1].reshape(current_batch_size, 2, -1)
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

        if epoch % 6 == 5 or test_only:
            model.eval()
            pbar = tqdm(dataloader_test, desc='Test')
            batch_index = 0
            loss = 0
            for batch in pbar:
                current_batch_size = batch[0].shape[0]
                y_flatten_exp = batch[0].reshape(current_batch_size, -1)
                y_flatten = torch.log10(y_flatten_exp + 1)

                x_flatten = batch[1].reshape(current_batch_size, 2, -1)
                y_pred = model(x_flatten)

                loss = criterion(y_pred, y_flatten)
                pbar.set_postfix({
                    'Epoch': epoch,
                    'Loss': loss.item(),
                    'Mean-A': model.a.mean().item(),
                    'Mean-B': model.b.mean().item(),
                    'Mean-C': model.c.mean().item()
                })

                writer.add_scalar('test_loss', loss.item(), epoch * len(dataloader_test) + batch_index)
                writer.add_scalar('test_mean_a', model.a.mean().item(), epoch * len(dataloader_test) + batch_index)
                writer.add_scalar('test_mean_b', model.b.mean().item(), epoch * len(dataloader_test) + batch_index)
                writer.add_scalar('test_mean_c', model.c.mean().item(), epoch * len(dataloader_test) + batch_index)

                if visualize:
                    writer.add_images('test_ln_y_pred_visual', y_pred.reshape(current_batch_size, 1, 256, 256),
                                      epoch * len(dataloader_test) + batch_index)
                    writer.add_images('test_ln_y_actual_visual', y_flatten.reshape(current_batch_size, 1, 256, 256),
                                      epoch * len(dataloader_test) + batch_index)

                    for bi in range(len(y_pred)):
                        visualize_array(y_pred[bi].reshape(256, 256).cpu().detach().numpy(), save=log_dir + "/test_ln_y_pred_visual", file=f'{epoch}_{batch_index}_{bi}.svg')
                        visualize_array(y_flatten[bi].reshape(256, 256).cpu().detach().numpy(), save=log_dir + "/test_ln_y_actual_visual", file=f'{epoch}_{batch_index}_{bi}.svg')

                    y_pred_visual = torch.pow(10, y_pred) - 1
                    y_pred_visual = y_pred_visual.reshape(current_batch_size, 1, 256, 256)
                    y_actual_visual = y_flatten_exp.reshape(current_batch_size, 1, 256, 256)

                    for bi in range(len(y_pred_visual)):
                        visualize_array(y_pred_visual[bi].reshape(256, 256).cpu().detach().numpy(), save=log_dir + "/test_y_pred_visual", file=f'{epoch}_{batch_index}_{bi}.svg')
                        visualize_array(y_actual_visual[bi].reshape(256, 256).cpu().detach().numpy(), save=log_dir + "/test_y_actual_visual", file=f'{epoch}_{batch_index}_{bi}.svg')

                    writer.add_images('test_y_pred_visual', y_pred_visual, epoch * len(dataloader_test) + batch_index)
                    writer.add_images('test_y_actual_visual', y_actual_visual,
                                      epoch * len(dataloader_test) + batch_index)

                batch_index += 1
            pbar.close()

            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_save_path = os.path.join(model_save_dir, f'model_{epoch}_{current_time}_{loss.item()}.pth')
            torch.save(model.state_dict(), model_save_path)

    # Print the optimized parameters
    print("Optimized parameters:", model.a, model.b, model.c)


if __name__ == '__main__':
    from fire import Fire

    Fire(_entry)
