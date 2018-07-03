import torch
import torch.nn as nn

class ShapenetShapeEncoder(nn.Module):
    def __init__(self):
        super(ShapenetShapeEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(4, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool3d(4)
        )
        self.outputs = nn.Linear(256 * 8 * 8 * 8, 128)

    def forward(self, inputs):
        conved = self.conv(inputs)
        outputs = self.outputs(conved.view(conved.size(0), -1))

        return outputs