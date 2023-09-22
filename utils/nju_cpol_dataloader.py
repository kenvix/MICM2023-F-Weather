import os
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


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
            self.padding_size = max([len(os.listdir(os.path.join(self.data_dir, self.dim_list[0], self.sub_dir, it))) for it in self.file_list])

    def __len__(self):
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
        results = np.vstack([[self._getitem_on_dim(idx, it) for it in self.dim_list]])
        return torch.tensor(results, device=self.device)

    @staticmethod
    def dataloader(data_dir, dim_list=None, sub_dir="1.0km", batch_size=16, shuffle=True, device='cpu', loader_args=None, **kwargs) -> DataLoader:
        if loader_args is None:
            loader_args = {}
        dataset = NjuCpolBaseDataset(data_dir, dim_list, sub_dir, device, **kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_args)
