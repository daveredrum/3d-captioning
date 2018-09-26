import torch
import torch.nn as nn
from lib.configs import CONF

class ShapeEncoder(nn.Module):
    def __init__(self, model_type, is_final=False):
        super(ShapeEncoder, self).__init__()
        if model_type == "noattention":
            self.conv = nn.Sequential(
                nn.Conv3d(4, 64, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(64),
                nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(128),
                nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(256),
                nn.Conv3d(256, 512, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(512)
            )
            self.outputs = nn.Linear(512, 128)
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(4, 64, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(64),
                nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(128),
                nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(256)
            )
            self.outputs = nn.Linear(256, 128)

        self.is_final = is_final

    def forward(self, inputs):
        conved = self.conv(inputs)
        pooled = conved.view(inputs.size(0), conved.size(1), -1).contiguous().mean(2)
        if self.is_final:
            outputs = (
                self.outputs(pooled.view(pooled.size(0), -1)),
                conved
            )
        else:
            outputs = self.outputs(pooled.view(pooled.size(0), -1))

        return outputs
