import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePaperNet(nn.Module):
    def __init__(self) -> None:
        super(BasePaperNet, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=(6, 6), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.dense1 = nn.Linear(46208, 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.dense1(out)

        return out
