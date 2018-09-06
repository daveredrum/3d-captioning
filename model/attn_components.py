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

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, ver=None):
        super(TemporalAttention, self).__init__()
        # basic settings
        self.hidden_size = hidden_size
        self.ver = ver
        # MLP
        if self.ver == "2.1-a":
            self.comp_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.comp_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
            self.comp_visual = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, 1, bias=False)
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)

    def forward(self, V, H):
        if self.ver == "2.1-a":
            H_t = H.permute(0, 2, 1).contiguous() # (batch_size, seq_size, hidden_size)
            H_mapped = self.comp_hidden(H_t) # (batch_size, seq_size, hidden_size)
            outputs = self.output_layer(F.tanh(H_mapped)) # (batch_size, seq_size, 1)
            outputs = F.softmax(outputs, dim=1).permute(0, 2, 1).contiguous() # (batch_size, 1, seq_size)
        else:
            H_t = H.permute(0, 2, 1).contiguous() # (batch_size, seq_size, hidden_size)
            V_t = V.permute(0, 2, 1).contiguous() # (batch_size, seq_size, hidden_size)
            H_mapped = self.comp_hidden(H_t) # (batch_size, seq_size, hidden_size)
            V_mapped = self.comp_visual(V_t) # (batch_size, seq_size, hidden_size)
            outputs = self.output_layer(F.tanh(V_mapped + H_mapped)) # (batch_size, seq_size, 1)
            outputs = F.softmax(outputs, dim=1).permute(0, 2, 1).contiguous() # (batch_size, 1, seq_size)

        return outputs


class AdaptiveTemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdaptiveTemporalAttention, self).__init__()
        # basic settings
        self.hidden_size = hidden_size
        # MLP
        # self.comp_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.comp_visual = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, 1, bias=False)
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)

    def forward(self, V, H, sentinel_scalar):
        H_t = H.permute(0, 2, 1).contiguous() # (batch_size, seq_size, hidden_size)
        V_t = V.permute(0, 2, 1).contiguous() # (batch_size, seq_size, hidden_size)
        # H_mapped = self.comp_hidden(H_t) # (batch_size, seq_size, hidden_size)
        # V_mapped = self.comp_visual(V_t) # (batch_size, seq_size, hidden_size)
        sentinel_scalar = sentinel_scalar.permute(0, 2, 1).contiguous() # (batch_size, seq_size, 1)
        balanced = (1 - sentinel_scalar) * V_t + sentinel_scalar * H_t # (batch_size, seq_size, hidden_size)
        outputs = self.output_layer(F.tanh(balanced)) # (batch_size, seq_size, 1)
        outputs = F.softmax(outputs, dim=1).permute(0, 2, 1).contiguous() # (batch_size, 1, seq_size)

        return outputs


class AdaptiveSpatialAttention(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(AdaptiveSpatialAttention, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.comp_visual = nn.Linear(visual_channels, hidden_size, bias=False)
        self.comp_hidden = nn.Linear(hidden_size, 1, bias=False)
        self.comp_sentinel = nn.Linear(hidden_size, 1, bias=False)
        self.output_layer = nn.Linear(hidden_size, 1, bias=False)
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, visual_inputs, states, sentinel):
        feature = visual_inputs.permute(0, 2, 1).contiguous() # (batch_size, visual_flat, visual_channels)
        # get the hidden state
        hidden = states[0] # (batch_size, hidden_size)
        V = self.comp_visual(feature) # (batch_size, visual_flat, hidden_size)
        H = self.comp_hidden(hidden).unsqueeze(1) # (batch_size, 1, 1)
        Z = F.tanh(V + H) # (batch_size, visual_flat, hidden_size)
        Z = self.output_layer(Z).squeeze(2) # (batch_size, visual_flat)
        attention_weights = F.softmax(Z, dim=1) # (batch_size, visual_flat)
        S = F.tanh(self.comp_sentinel(sentinel) + self.comp_hidden(hidden)) # (batch_size, 1)
        sentinel_scalar = F.softmax(torch.cat((Z, S), dim=1), dim=1)[:, -1].unsqueeze(1) # (batch_size, 1)

        return attention_weights, sentinel_scalar

# 3D attention module
class Attention3D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(Attention3D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.comp_visual = nn.Linear(visual_channels, hidden_size, bias=False)
        self.comp_hidden = nn.Linear(hidden_size, 1, bias=False)
        self.output_layer = nn.Linear(hidden_size, 1, bias=False)
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
        # combine
        Z = F.tanh(V + H)
        # attention_weights = (batch_size, visual_flat)
        attention_weights = self.output_layer(Z).squeeze(2)
        attention_weights = F.softmax(attention_weights, dim=1)

        return attention_weights


# 3D adaptive attention module
class AdaptiveAttention3D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(AdaptiveAttention3D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.comp_visual = nn.Linear(visual_channels, hidden_size, bias=False)
        self.comp_hidden = nn.Linear(hidden_size, 1, bias=False)
        self.comp_sentinel = nn.Linear(hidden_size, 1, bias=False)
        self.output_layer = nn.Linear(hidden_size, 1, bias=False)
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, visual_inputs, states, sentinel):
        feature = visual_inputs.permute(0, 2, 1).contiguous() # (batch_size, visual_flat, visual_channels)
        hidden = states[0] # (batch_size, hidden_size)
        V = self.comp_visual(feature) # (batch_size, visual_flat, hidden_size)
        H = self.comp_hidden(hidden).unsqueeze(1) # (batch_size, 1, 1)
        # combine
        Z = F.tanh(V + H)
        Z = self.output_layer(Z).squeeze(2) # (batch_size, visual_flat)
        # sentinel outputs
        S = F.tanh(self.comp_sentinel(sentinel) + self.comp_hidden(hidden)) # (batch_size, 1)
        # concat
        outputs = torch.cat((Z, S), dim=1)
        outputs = F.softmax(outputs, dim=1) # (batch_size, visual_flat + 1)

        return outputs

# self-attention module
class SelfAttention3D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(SelfAttention3D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.f = nn.Linear(visual_channels, hidden_size, bias=False)
        self.g = nn.Linear(visual_channels, hidden_size, bias=False)
        self.h = nn.Linear(visual_channels, visual_channels, bias=False)
        self.output_layer = nn.Linear(visual_flat, 1, bias=False)
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)

    def forward(self, visual_inputs):
        feature = visual_inputs.permute(0, 2, 1).contiguous() # (batch_size, visual_flat, visual_channels)
        f = self.f(feature) # (batch_size, visual_flat, hidden_size)
        g = self.g(feature) # (batch_size, visual_flat, hidden_size)
        h = self.h(feature).transpose(2, 1).contiguous() # (batch_size, visual_channels, visual_flat)
        s = f.matmul(g.transpose(2, 1).contiguous()) # (batch_size, visual_flat, visual_flat)
        s_comp = self.output_layer(s) # (batch_size, visual_flat, 1)
        mask = F.softmax(s_comp, dim=0) # (batch_size, visual_flat, 1)
        outputs = h.matmul(mask).squeeze(2) # (batch_size, visual_channels)

        return outputs, mask

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalSelfAttention, self).__init__()
        # basic settings
        self.hidden_size = hidden_size
        # MLP
        self.comp_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, 1, bias=False)
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        h_comp = self.comp_h(h) # (batch_size, seq_size, hidden_size)
        outputs = self.output_layer(h_comp) # (batch_size, seq_size, 1)

        return outputs