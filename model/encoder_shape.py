import torch
import torch.nn as nn

class ShapenetShapeEncoder(nn.Module):
    def __init__(self):
        super(ShapenetShapeEncoder, self).__init__()
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
        self.outputs = nn.Linear(256, 128)

    def forward(self, inputs):
        conved = self.conv(inputs)
        conved = conved.view(inputs.size(0), conved.size(1), -1).contiguous().mean(2)
        outputs = self.outputs(conved.view(conved.size(0), -1))

        return outputs
