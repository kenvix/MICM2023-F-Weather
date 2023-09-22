import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NjuCpolBaseDataset(Dataset):
    def __init__(self, data_dir, device='cpu', **kwargs):
        self.data_dir = data_dir
        self.device = device
        self.padding = kwargs.get('padding', False)
        self.padding_size = int(kwargs.get('padding_size', -1))

        self.file_list = os.listdir(data_dir)
        if len(self.file_list) == 0:
            raise RuntimeError(f"Empty directory: {data_dir}")

        # Inspection
        self.frame_shape = os.path.join(self.data_dir, self.file_list[0])
        self.frame_shape = os.path.join(self.frame_shape, os.listdir(self.frame_shape)[0])
        self.frame_shape = np.load(self.frame_shape).shape

        if self.padding and self.padding_size == -1:
            # auto detect max length of files
            self.padding_size = max([len(os.listdir(os.path.join(self.data_dir, it))) for it in self.file_list])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> torch.Tensor:
        dir_file_path = os.path.join(self.data_dir, self.file_list[idx])
        frame_file_list = os.listdir(dir_file_path)
        results = [np.load(os.path.join(dir_file_path, it)) for it in frame_file_list]
        v = [results]
        if self.padding:
            if len(results) < self.padding_size:
                paddings = np.zeros([self.padding_size - len(results), *self.frame_shape])
                v.append(paddings)

        results = np.vstack(v)
        return torch.tensor(results, device=self.device)

    @staticmethod
    def dataloader(data_dir, batch_size=16, shuffle=True, device='cpu', loader_args=None, **kwargs) -> DataLoader:
        if loader_args is None:
            loader_args = {}
        dataset = NjuCpolBaseDataset(data_dir, device, **kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_args)
