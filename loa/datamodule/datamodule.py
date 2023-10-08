import os
from typing import Optional

import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import Dataset, DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.image_name_list = os.listdir(f"{file_path}/images")
        self.data_len = self.image_name_list.__len__()

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        image = cv2.imread(f"{self.file_path}/images/{item}.png")
        if np.shape(image)[0] != 512:
            image = cv2.resize(image, (512, 512))
        label = cv2.imread(f"{self.file_path}/labels/{item}.png", cv2.IMREAD_GRAYSCALE)
        if np.shape(label)[0] != 512:
            label = cv2.resize(label, (512, 512))
        label = label.astype(np.longlong)
        image = transforms.ToTensor()(image)
        label = torch.from_numpy(label)
        data = (image, label)
        return data


class LOADatamodule(pl.LightningDataModule):
    def __init__(
            self,
            data_type: str,
            batch_size: int = 8,
            num_workers: int = 16,
            drop_last: bool = False
    ) -> None:
        super().__init__()
        dataset_path = f"{os.path.dirname(os.path.realpath(__file__))}/../../dataset/{data_type}"
        self.train_path = f"{dataset_path}/train"
        self.val_path = f"{dataset_path}/val"
        self.test_path = f"{dataset_path}/test"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_file_path = self.train_path
            train_dataset = MyDataset(train_file_path)
            print("\033[0;31;40m train_dataset_len = ", len(train_dataset), "\033[0m")
            self.train_dataset = train_dataset

            val_file_path = self.val_path
            val_dataset = MyDataset(val_file_path)
            print("\033[0;31;40m val_dataset_len = ", len(val_dataset), "\033[0m")
            self.val_dataset = val_dataset
        if stage == "test" or stage is None:
            test_file_path = self.test_path
            test_dataset = MyDataset(test_file_path)
            print("\033[0;31;40m test_dataset_len = ", len(test_dataset), "\033[0m")
            self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers
        )
