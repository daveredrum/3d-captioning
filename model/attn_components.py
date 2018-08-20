import math
import torch
import copy
import numpy as np
import torch.nn as nn
from collections import deque
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as torchmodels
import torch.nn.functional as F
from torch.nn import Parameter

# attention module
class Attention3D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(Attention3D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.comp_visual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.comp_hidden = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, visual_inputs, states):
        # visual_inputs = (batch_size, visual_flat, visual_channels)
        feature = visual_inputs.permute(0, 2, 1).contiguous()
        # get the hidden state
        hidden = states[0]
        # in = (batch_size, visual_flat, visual_channels)
        # out = (batch_size, visual_flat, hidden_size)
        V = self.comp_visual(feature)
        # in = (batch_size, hidden_size)
        # out = (batch_size, 1, hidden_size)
        H = self.comp_hidden(hidden).unsqueeze(1)
        # print("V", V.view(-1).min(0)[0].item(), V.view(-1).max(0)[0].item())
        # print("H", H.view(-1).min(0)[0].item(), H.view(-1).max(0)[0].item())
        # combine
        outputs = F.tanh(V + H)
        # outputs = (batch_size, visual_flat)
        outputs = self.output_layer(outputs).squeeze(2)
        outputs = F.softmax(outputs, dim=1)

        return outputs


# adaptive attention module
class AdaptiveAttention3D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(AdaptiveAttention3D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.comp_visual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.comp_hidden = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.comp_sentinel = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, visual_inputs, states, sentinel):
        # visual_inputs = (batch_size, visual_flat, visual_channels)
        feature = visual_inputs.permute(0, 2, 1).contiguous()
        # get the hidden state
        hidden = states[0]
        # in = (batch_size, visual_flat, visual_channels)
        # out = (batch_size, visual_flat, hidden_size)
        V = self.comp_visual(feature)
        # in = (batch_size, hidden_size)
        # out = (batch_size, 1, 1)
        H = self.comp_hidden(hidden).unsqueeze(1)
        # combine
        Z = F.tanh(V + H)
        # Z = (batch_size, visual_flat)
        Z = self.output_layer(Z).squeeze(2)
        # sentinel outputs
        # S = (batch_size, 1)
        S = F.tanh(self.comp_sentinel(sentinel) + self.comp_hidden(hidden))
        # concat
        # outputs = (batch_size, visual_flat + 1)
        outputs = torch.cat((Z, S), dim=1)
        outputs = F.softmax(outputs, dim=1)

        return outputs