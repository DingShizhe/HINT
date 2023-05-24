from functools import partial
import os
import torch
import scipy.io as sio
import numpy as np

import pytorch_lightning as pl
import pdb
from noise import pnoise2


class PerlinDataset(torch.utils.data.Dataset):
    def __init__(self, num=1024, p_num=256, test=False):
        super(PerlinDataset, self).__init__()

        if test:
            self.seeds = np.arange(0, 1024)
        else:
            self.seeds = np.random.randint(1024, 2**16 - 1, (num))

        if test:
            # self.x = np.loadtxt("tensor_x.txt")
            self.x = None
        else:
            self.x = None

        self.dim = 2
        self.support = [-1, 1]
        self.p_num = p_num


        self.test = test
        
    def __len__(self):
        return len(self.seeds)
    
    @classmethod
    def pnoise2_func_given_seed(cls, seed, scale = 0.5, octaves = 3, persistence = 0.5, lacunarity = 2.0):

        return lambda x0, x1: pnoise2(x0 * scale, x1 * scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    base=seed
        )

    def __getitem__(self, index):


        if self.x is None:
            x = np.random.rand(self.p_num, self.dim) * (self.support[1]-self.support[0]) + self.support[0]
        else:
            x = self.x

        seed = self.seeds[index]

        scale = 0.5
        octaves = 3
        persistence = 0.5
        lacunarity = 2.0

        _pnoise2_func = self.pnoise2_func_given_seed(seed)

        y = []
        for _x in x:
            _y = _pnoise2_func(_x[0], _x[1])
            y.append(_y)

        y = np.array(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)

        return torch.cat([x, y[...,None]], axis=1).float()


def _collate(data):
    data = torch.stack(data)
    O, T = data[:, :64, :], data[:, 64:, :]
    return O[...,0:-1], O[...,-1:], T[...,0:-1], T[...,-1:], 64


class PerlinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        train_path,
        test_path,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_path = train_path
        self.test_path = test_path

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.train_dataset = PerlinDataset(1024 * 256)
        self.test_dataset = PerlinDataset(512, test=True)

    def setup(self, stage=None):
        self.prepare_data()

    def __dataloader(self, dataset, shuffle=False):
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            collate_fn=partial(_collate),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return loader

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False)
