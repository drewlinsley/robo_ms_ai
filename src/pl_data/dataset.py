from omegaconf import DictConfig, ValueNode
import torch
from torch.utils.data import Dataset
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import csv
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10 as cifar10_data
from torch.nn import functional as F
import numpy as np
from glob2 import glob
from skimage import io


DATADIR = "data/"


def load_image(directory):
    return Image.open(directory).convert('L')


def invert(img):

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))

    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img


def colour(img, ch=0, num_ch=3):

    colimg = [torch.zeros_like(img)] * num_ch
    # colimg[ch] = img
    # Use beta distribution to push the mixture to ch 1 or ch 2
    if ch == 0:
        rand = torch.distributions.beta.Beta(0.5, 1.)
    elif ch == 1:
        rand = torch.distributions.beta.Beta(1., 0.5) 
    else:
        raise NotImplementedError("Only 2 channel images supported now.")
    rand = rand.sample()
    colimg[0] = img * rand
    colimg[1] = img * (1 - rand)
    return torch.cat(colimg)


class CIFAR10(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform

        self.dataset = cifar10_data(root=DATADIR, download=True)
        self.data_len = len(self.dataset)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        img, label = self.dataset[index]
        img = np.asarray(img)
        label = np.asarray(label)
        # Transpose shape from H,W,C to C,H,W
        img = img.transpose(2, 0, 1).astype(np.float32)
        # img = F.to_tensor(img)
        # label = F.to_tensor(label)
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class COR14(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval

        # List all the files
        print("Globbing files for COR14, this may take a while...")
        self.files = glob(os.path.join(self.path, "**", "**", "*.tif"))
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        img = self.files[index]
        img = io.imread(img, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        label = 0  # Set a fixed label for now. Dummy.
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class NuclearGedi(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval

        curated = True if ".npz" in path else False
        if curated:
            data = np.load(path)
            files = data["files"]
            labels = data["labels"]
            # files = np.asarray(data["files"]).reshape(-1, 1)
            # labels = np.asarray(data["labels"]).reshape(-1, 1)
            # files = torch.from_numpy(files)
            labels = torch.from_numpy(labels)
        else:
            # List all the files
            print("Globbing files for NuclearGedi, this may take a while...")
            live = glob(os.path.join(self.path, "GC150nls-Live", "*.tif"))
            dead = glob(os.path.join(self.path, "GC150nls-Dead", "*.tif"))
            if not len(live) or not len(dead):
                raise RuntimeError("No files found at {}".format(self.path))
            files = np.asarray(live + dead)
            labels = np.concatenate((np.ones(len(live)), np.zeros(len(dead))), 0).astype(int).reshape(-1, 1)
        np.random.seed(42)
        idx = np.random.permutation(len(files))
        files = files[idx]  # Shuffle
        labels = labels[idx]
        self.files = files
        self.labels = labels
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        label = self.labels[index]
        label = label.reshape(-1)
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        return img, label, fn

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class NonNuclearGedi(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval

        curated = True if ".npz" in path else False
        if curated:
            data = np.load(path)
            files = data["files"]
            labels = data["labels"]
            # files = np.asarray(data["files"]).reshape(-1, 1)
            # labels = np.asarray(data["labels"]).reshape(-1, 1)
            # files = torch.from_numpy(files)
            labels = torch.from_numpy(labels)
        else:
            # List all the files
            print("Globbing files for NuclearGedi, this may take a while...")
            live = glob(os.path.join(self.path, "GC150nls-Live", "*.tif"))
            dead = glob(os.path.join(self.path, "GC150nls-Dead", "*.tif"))
            if not len(live) or not len(dead):
                raise RuntimeError("No files found at {}".format(self.path))
            files = np.asarray(live + dead)
            labels = np.concatenate((np.ones(len(live)), np.zeros(len(dead))), 0).astype(int).reshape(-1, 1)
        np.random.seed(42)
        idx = np.random.permutation(len(files))
        files = files[idx]  # Shuffle
        labels = labels[idx]
        self.files = files
        self.labels = labels
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        label = self.labels[index]
        label = label.reshape(-1)
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel
        return img, label, fn

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"

