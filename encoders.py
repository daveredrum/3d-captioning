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

class EncoderResNet101(nn.Module):
    def __init__(self):
        super(EncoderResNet101, self).__init__()
        resnet = torchmodels.resnet101(pretrained=True)
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
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_layer = nn.Sequential(
            *list(vgg16.classifier.children())[:-1]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.5)
        )
        
    
    def forward(self, inputs):
        '''
        original_features: (batch_size, 512, 14, 14)
        outputs: (batch_size, 512)
        '''
        batch_size = inputs.size(0)
        original_features = inputs.view(inputs.size(0), 512, 14, 14)
        outputs = self.max_pool(original_features).view(batch_size, -1)
        outputs = self.fc_layer(outputs)
        outputs = self.output_layer(outputs)

        return outputs

# for attention
class AttentionResNet101(nn.Module):
    def __init__(self):
        super(AttentionResNet101, self).__init__()
        resnet = torchmodels.resnet101(pretrained=True)
        self.resnet = nn.Sequential(
            *list(resnet.children())[:-2],
        )


    def forward(self, inputs):
        '''
        original_features: (batch_size, 2048 * 7 * 7)
        '''
        original_features = self.resnet(inputs)
        original_features = original_features.view(original_features.size(0), -1)

        return original_features

# for attention
class AttentionEncoderResNet101(nn.Module):
    def __init__(self):
        super(AttentionEncoderResNet101, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.global_mapping = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.area_mapping = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )


    def forward(self, inputs):
        '''
        original_features: (batch_size, 2048, 7, 7)
        global_features: (batch_size, 512)
        area_features: (batch_size, 512, 49)
        '''
        batch_size = inputs.size(0)
        original_features = inputs.view(inputs.size(0), 2048, 7, 7)
        # (batch_size, 512, 49)
        area_features = original_features.view(batch_size, 2048, -1).transpose(2, 1).contiguous()
        area_features = self.area_mapping(area_features).transpose(2, 1).contiguous().view(batch_size, 512, -1)
        # (batch_size, 512)
        global_features = self.avg_pool(original_features).view(batch_size, 2048)
        global_features = self.global_mapping(global_features)

        return original_features, global_features, area_features

# for attention
class AttentionVGG16BN(nn.Module):
    def __init__(self):
        super(AttentionVGG16BN, self).__init__()
        vgg16 = torchmodels.vgg16_bn(pretrained=True)
        self.vgg16 = nn.Sequential(
            *list(vgg16.features.children())[:-1],
        )


    def forward(self, inputs):
        '''
        original_features: (batch_size, 512 * 14 * 14)
        '''
        original_features = self.vgg16(inputs)
        original_features = original_features.view(original_features.size(0), -1)

        return original_features

# for attention
class AttentionEncoderVGG16BN(nn.Module):
    def __init__(self):
        super(AttentionEncoderVGG16BN, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=14, stride=14)
        self.global_mapping = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.5)
        )
        self.area_mapping = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.5)
        )


    def forward(self, inputs):
        '''
        original_features: (batch_size, 512, 14, 14)
        global_features: (batch_size, 512)
        area_features: (batch_size, 512, 196)
        '''
        batch_size = inputs.size(0)
        original_features = inputs.view(inputs.size(0), 512, 14, 14)
        # (batch_size, 512, 196)
        area_features = original_features.view(batch_size, 512, -1).transpose(2, 1).contiguous()
        area_features = self.area_mapping(area_features).transpose(2, 1).contiguous().view(batch_size, 512, -1)
        # (batch_size, 512)
        global_features = self.avg_pool(original_features).view(batch_size, 512)
        global_features = self.global_mapping(global_features)

        return original_features, global_features, area_features


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

