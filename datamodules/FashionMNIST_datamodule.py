import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
import torch
import os

DATA_PATH = os.path.join(os.getcwd(), 'data')
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
mean = 0.3
std = 0.3

class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir=DATA_PATH):
        self.data_dir = data_dir
        super().__init__()

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        )
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            fashionmnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = random_split(fashionmnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCH_SIZE)