from typing import Optional, Sequence

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, ValueNode
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from src.pl_data import normalizations
from PIL import Image
from importlib import import_module
from src.pl_data import dataset
from torch import default_generator, Generator
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from itertools import accumulate as _accumulate
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform, SimCLRTrainDataTransform)


from typing import (
    TypeVar,
    List,
)

T = TypeVar('T')


TRANSFORM_RECIPES = {
    "CIFAR10": {
        "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                normalizations.cifar10_normalization(),
            ]),
        "val": transforms.Compose([
                transforms.ToTensor(),
                normalizations.cifar10_normalization(),
            ]),
        "test": transforms.Compose([
                transforms.ToTensor(),
                normalizations.cifar10_normalization(),
            ]),
    },
    "COR14": {
        "train": transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                normalizations.COR14_normalization(),
            ]),
        "val": transforms.Compose([
                transforms.ToTensor(),
                normalizations.COR14_normalization(),
            ]),
        "test": transforms.Compose([
                transforms.ToTensor(),
                normalizations.COR14_normalization(),
            ]),
    },
    "SIMCLR_COR14": {
        "train": SimCLRTrainDataTransform(
            input_height=224,
            gaussian_blur=True,
            jitter_strength=1.,
            normalize=normalizations.COR14_normalization),  # noqa
        "val": SimCLREvalDataTransform(
            input_height=224,
            gaussian_blur=False,
            jitter_strength=0.,
            normalize=normalizations.COR14_normalization),  # noqa
        "test": SimCLREvalDataTransform(
            input_height=224,
            gaussian_blur=False,
            jitter_strength=0.,
            normalize=normalizations.COR14_normalization),  # noqa
    },
}


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        val_proportion: float,
        cfg: DictConfig,
        dataset_name: str,
        transform_recipe: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_proportion = val_proportion
        self.dataset_name = dataset_name
        self.datasets = datasets
        self.transform_recipe = transform_recipe

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None
        self.dataset = getattr(dataset, self.dataset_name)


    def setup(self, stage: Optional[str] = None):
        """Prepare datasets for train/val/test."""
        selected_recipe = TRANSFORM_RECIPES.get(self.transform_recipe, None)
        assert transforms is not None, "Could not recognize your transform recipe: {}".format(self.transform_recipe)  # noqa
        train_transform = selected_recipe["train"]
        val_transform = selected_recipe["val"]
        test_transform = selected_recipe["test"]

        # split dataset
        if stage is None or stage == "fit":
            assert self.val_proportion >= 0 and self.val_proportion < 1., \
                "val_proportion must be 1 > x >= 0"
            plank_train = hydra.utils.instantiate(
                self.datasets[self.dataset_name].train,
                cfg=self.cfg,
                transform=train_transform,
                _recursive_=False
            )
            if self.val_proportion == 0:
                plank_val = hydra.utils.instantiate(
                    self.datasets[self.dataset_name].val,
                    cfg=self.cfg,
                    transform=val_transform,
                    _recursive_=False
                )
                self.train_dataset = plank_train
                self.val_dataset = plank_val
            else:
                train_length = int(len(plank_train) * (1 - self.val_proportion))  # noqa
                val_length = len(plank_train) - train_length
                self.train_dataset, self.val_dataset = continuous_random_split(
                    plank_train, [train_length, val_length]
                )

        elif stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(
                    self.datasets[self.dataset_name].test,
                    cfg=self.cfg,
                    transform=test_transform,
                    _recursive_=False)
            ]
        else:
            raise NotImplementedError("stage: {}".format(stage))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            multiprocessing_context='fork'
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            multiprocessing_context='fork'
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                multiprocessing_context='fork'
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets}, "
            f"{self.num_workers}, "
            f"{self.batch_size})"
            f"{self.val_proportion}"
        )



def continuous_random_split(dataset: Dataset[T], lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = torch.arange(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
