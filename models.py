import torch
import torch.nn as nn
from torch.autograd import Variable

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 58 * 58, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        outputs = self.conv_layer(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        return outputs