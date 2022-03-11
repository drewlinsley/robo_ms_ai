import math
from argparse import ArgumentParser

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
# from pl_bolts.models.self_supervised import SimCLR


def resnet18(pretrained=False, num_classes=None, num_samples=None, batch_size=None):
    assert num_classes is not None, "You must pass the number of classes to your model."
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


def simclr_resnet18(pretrained=False, num_classes=None, num_samples=None, batch_size=None):
    assert num_samples is not None, "You must pass the number of samples to the SimCLR class."
    assert batch_size is not None, "You must pass the batch size to the SimCLR class."
    model = SimCLR(num_samples=num_samples, batch_size=batch_size, arch="resnet18")
    return model


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(LightningModule):
    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        num_nodes: int = 1,
        arch: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        first_conv: bool = True,
        maxpool1: bool = True,
        exclude_bn_bias: bool = False,
        weight_decay: float = 1e-6,
    ):

        self.num_nodes = num_nodes
        self.arch = arch
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay

        self.encoder = self.init_model()

        self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50
        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def forward(self, x):
        # bolts resnet returns a list
        return self.encoder(x)[-1]

    def shared_step(self, batch):

        # final image in tuple is for online eval
        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        return z1, z2
