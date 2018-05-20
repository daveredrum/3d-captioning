import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as torchmodels
import torch.nn.functional as F
from torch.nn import Parameter

class Encoder2D(nn.Module):
    def __init__(self):
        super(Encoder2D, self).__init__()
        self.conv_layer = nn.Sequential(
            # nn.Conv2d(3, 16, 3),
            # nn.ReLU(),
            # nn.BatchNorm2d(16),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(16, 32, 3),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(32, 64, 3),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.MaxPool2d(2, 2)
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            # nn.Linear(64 * 14 * 14, 512),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        outputs = self.conv_layer(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        outputs = self.output_layer(outputs)
        
        return outputs

    # chop the last output layer
    def extract(self, inputs):
        outputs = self.conv_layer(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        
        return outputs

class EncoderResnet50(nn.Module):
    def __init__(self):
        super(EncoderResnet50, self).__init__()
        resnet = torchmodels.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc_layer = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
        
    
    def forward(self, inputs):
        outputs = self.resnet(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        outputs = self.output_layer(outputs)
        
        return outputs

    # chop the last output layer
    def extract(self, inputs):
        outputs = self.resnet(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        
        return outputs

class EncoderVGG16(nn.Module):
    def __init__(self):
        super(EncoderVGG16, self).__init__()
        vgg16 = torchmodels.vgg16(pretrained=True)
        self.vgg16 = vgg16.features
        self.fc_layer = nn.Sequential(
            *list(vgg16.classifier.children())[:-1],
            nn.Linear(vgg16.classifier[6].in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
        
    
    def forward(self, inputs):
        outputs = self.vgg16(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        outputs = self.output_layer(outputs)
        
        return outputs

    # chop the last output layer
    def extract(self, inputs):
        outputs = self.vgg16(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        
        return outputs

class EncoderVGG16BN(nn.Module):
    def __init__(self):
        super(EncoderVGG16BN, self).__init__()
        vgg16 = torchmodels.vgg16_bn(pretrained=True)
        self.vgg16 = vgg16.features
        self.fc_layer = nn.Sequential(
            *list(vgg16.classifier.children())[:-1],
            nn.Linear(vgg16.classifier[6].in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
        
    
    def forward(self, inputs):
        outputs = self.vgg16(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        outputs = self.output_layer(outputs)
        
        return outputs

    # chop the last output layer
    def extract(self, inputs):
        outputs = self.vgg16(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        
        return outputs

# for attention
class AttentionEncoderVGG16(nn.Module):
    def __init__(self):
        super(AttentionEncoderVGG16, self).__init__()
        vgg16 = torchmodels.vgg16(pretrained=True)
        self.vgg16 = nn.Sequential(
            *list(vgg16.features.children())[:-1],
        )


    def forward(self, inputs):
        outputs = self.vgg16(inputs)
        
        return outputs

# for attention
class AttentionEncoderVGG16BN(nn.Module):
    def __init__(self):
        super(AttentionEncoderVGG16BN, self).__init__()
        vgg16 = torchmodels.vgg16_bn(pretrained=True)
        self.vgg16 = nn.Sequential(
            *list(vgg16.features.children())[:-1],
        )    


    def forward(self, inputs):
        outputs = self.vgg16(inputs)
        
        return outputs

# for attention
class AttentionEncoderResnet50(nn.Module):
    def __init__(self):
        super(AttentionEncoderResnet50, self).__init__()
        resnet = torchmodels.resnet50(pretrained=True)
        self.resnet = nn.Sequential(
            *list(resnet.children())[:-2],
            # (2048, 7, 7)
        )
        
    
    def forward(self, inputs):
        outputs = self.resnet(inputs)
        
        return outputs


class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv3d(3, 16, 3),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 2 * 2 * 2, 512), # for 32
            # nn.Linear(64 * 14 * 14 * 14, 512), # for 128
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        outputs = self.conv_layer(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        outputs = self.output_layer(outputs)
        
        return outputs

    # chop the last output layer
    def extract(self, inputs):
        outputs = self.conv_layer(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        
        return outputs

