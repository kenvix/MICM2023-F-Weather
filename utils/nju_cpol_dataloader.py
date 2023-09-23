import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

norm_param = {
    'dBZ': [0, 65],
    'ZDR': [-1, 5],
    'KDP': [-1, 6]
}


def normalize_radar(x: np.ndarray, target='dBZ'):
    mmin, mmax = norm_param[target]
    data_frame = (x - mmin) / (mmax - mmin)
    return data_frame


class NjuCpolBaseDataset(Dataset):
    def __init__(self, data_dir, dim_list=None, sub_dir="1.0km", device='cpu', **kwargs):
        self.sub_dir = sub_dir
        if dim_list is None:
            dim_list = ['dBZ', 'KDP', 'ZDR']

        self.dim_list = dim_list
        self.data_dir = data_dir
        self.device = device
        self.padding = kwargs.get('padding', False)
        self.padding_size = int(kwargs.get('padding_size', -1))

        # Inspection
        inspection_dir = os.path.join(self.data_dir, self.dim_list[0], self.sub_dir)
        self.file_list = os.listdir(inspection_dir)

        if len(inspection_dir) == 0:
            raise RuntimeError(f"Empty directory: {inspection_dir}")

        self.frame_shape = os.path.join(inspection_dir, self.file_list[0])
        self.frame_shape = os.path.join(self.frame_shape, os.listdir(self.frame_shape)[0])
        self.frame_shape = np.load(self.frame_shape).shape

        if self.padding and self.padding_size == -1:
            # auto detect max length of files
            self.padding_size = max(
                [len(os.listdir(os.path.join(self.data_dir, self.dim_list[0], self.sub_dir, it))) for it in
                 self.file_list])

    def __len__(self) -> int:
        return len(self.file_list)

    def _getitem_on_dim(self, idx, dim) -> np.ndarray:
        dir_file_path = os.path.join(self.data_dir, dim, self.sub_dir, self.file_list[idx])
        frame_file_list = os.listdir(dir_file_path)
        results = [np.load(os.path.join(dir_file_path, it)) for it in frame_file_list]
        v = [results]
        if self.padding:
            if len(results) < self.padding_size:
                paddings = np.zeros([self.padding_size - len(results), *self.frame_shape])
                v.append(paddings)

        return np.vstack(v)

    def __getitem__(self, idx) -> torch.Tensor:
        results = np.stack([self._getitem_on_dim(idx, it) for it in self.dim_list], axis=0)
        return torch.tensor(results, device=self.device)

    @staticmethod
    def dataloader(data_dir, dim_list=None, sub_dir="1.0km", batch_size=16, window_size=-1, window_step=1, shuffle=True,
                   device='cpu', loader_args=None, **kwargs) -> DataLoader:
        if loader_args is None:
            loader_args = {}
        dataset = NjuCpolBaseDataset(data_dir, dim_list, sub_dir, device, **kwargs)
        if window_size > 0:
            dataset = WindowedDataset(dataset, window_size, window_step, device)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_args)


class NjuCpolFrameWindowedDataset(Dataset):
    def __init__(self, data_dir, window_size_in=(2, 10), dim_list=None, sub_dir="1.0km", norm=True, device='cpu', **kwargs):
        if dim_list is None:
            dim_list = ['dBZ', 'KDP', 'ZDR']
        self.dim_list = dim_list
        self.norm = norm
        self.sub_dir = sub_dir
        self.data_dir = data_dir
        self.device = device
        self.window_size_in = window_size_in
        self.window_size_per_dim = window_size_in[1]
        self.window_size = window_size_in[0]

        # Inspection
        inspection_dir = os.path.join(self.data_dir, self.dim_list[0], self.sub_dir)
        self.file_list = os.listdir(inspection_dir)

        if len(inspection_dir) == 0:
            raise RuntimeError(f"Empty directory: {inspection_dir}")

        self.frame_shape = os.path.join(inspection_dir, self.file_list[0])
        self.frame_shape = os.path.join(self.frame_shape, os.listdir(self.frame_shape)[0])
        self.frame_shape = np.load(self.frame_shape).shape

        file_count = 0
        directory_count = 0
        self.total_length = 0
        self.item_list = []
        for root, dirs, files in os.walk(inspection_dir):
            file_count += len(files)
            directory_count += len(dirs)
            window = len(files) - self.window_size_per_dim + 1
            self.total_length += window
            for i in range(window):
                self.item_list.append([os.path.join(root, files[j]) for j in range(i, i + self.window_size_per_dim)])

        self.file_count = file_count

    def __len__(self) -> int:
        return len(self.item_list)

    def __getitem__(self, idx) -> torch.Tensor:
        results = [np.load(it) for it in self.item_list[idx]]
        v = np.stack(results, axis=0)
        return torch.tensor(v, device=self.device)

    @staticmethod
    def dataloader(data_dir, dim='dBZ', sub_dir="1.0km", norm=True, batch_size=16, window_size=(2, 10), shuffle=True,
                   device='cpu', loader_args=None, **kwargs) -> DataLoader:
        if loader_args is None:
            loader_args = {}
        dataset = NjuCpolFrameWindowedDataset(data_dir, dim=dim, window_size_in=window_size, sub_dir=sub_dir,
                                              device=device, norm=norm, **kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_args)


class NjuCpolCoupledDataset(Dataset):
    def __init__(self, data_dir, label_dir="kdp-rain", dim_list=None, sub_dir="1.0km", norm=True, device='cpu',
                 **kwargs):
        self.norm = norm
        self.label_dir = label_dir
        self.sub_dir = sub_dir
        if dim_list is None:
            dim_list = ['dBZ', 'ZDR']

        self.dim_list = dim_list
        self.data_dir = data_dir
        self.device = device

        # Inspection
        inspection_dir = os.path.join(self.data_dir, self.dim_list[0], self.sub_dir)

        self.file_list = os.listdir(inspection_dir)

        if len(inspection_dir) == 0:
            raise RuntimeError(f"Empty directory: {inspection_dir}")

        self.frame_shape = os.path.join(inspection_dir, self.file_list[0])
        self.frame_shape = os.path.join(self.frame_shape, os.listdir(self.frame_shape)[0])
        self.frame_shape = np.load(self.frame_shape).shape

        label_path = os.path.join(self.data_dir, self.label_dir)
        file_count = 0
        directory_count = 0
        self.total_length = 0
        self.item_list = []
        for root, dirs, files in os.walk(label_path):
            file_count += len(files)
            directory_count += len(dirs)
            dir_name = os.path.basename(root)
            for file in files:
                self.item_list.append([
                    os.path.join(root, file),
                    *(os.path.join(self.data_dir, dim, sub_dir, dir_name, file) for dim in self.dim_list),
                ])

        self.file_count = file_count

    def __len__(self) -> int:
        return len(self.item_list)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        results = [np.load(it) for it in self.item_list[idx]]
        if self.norm:
            for i in range(len(self.dim_list)):
                results[i] = normalize_radar(results[i], target=self.dim_list[i])
        v = np.stack(results[1:], axis=0)
        return torch.tensor(results[0], device=self.device), torch.tensor(v, device=self.device)

    @staticmethod
    def dataloader(data_dir, label_dir="kdp-rain", dim_list=None, sub_dir="1.0km", norm=True, batch_size=16,
                   shuffle=True, device='cpu', loader_args=None, **kwargs) -> DataLoader:
        if loader_args is None:
            loader_args = {}
        dataset = NjuCpolCoupledDataset(data_dir, label_dir=label_dir, dim_list=dim_list, sub_dir=sub_dir,
                                        device=device, norm=norm, **kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_args)


class WindowedDataset(Dataset):
    def __init__(self, data: Dataset, window_size, window_step, device='cpu'):
        self.window_step = window_step
        self.device = device
        self.data = data
        self.window_size = window_size

    def __getitem__(self, index):
        w = [self.data[index + i] for i in range(self.window_size + 1)]
        x = torch.stack(w, dim=0)
        return x

    def __len__(self):
        # noinspection PyTypeChecker
        # length is depended on window size and step
        return len(self.data) - self.window_size + 1
