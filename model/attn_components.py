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
from lib.configs import CONF


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


# self-attention module
class SelfAttention3D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat, attention_type):
        super(SelfAttention3D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        self.attention_type = attention_type
        # MLP
        if self.attention_type == 'self-sep' or self.attention_type == 'self-nosep':
            self.comp_f = nn.Linear(visual_channels, hidden_size, bias=False)
            self.comp_g = nn.Linear(visual_channels, hidden_size, bias=False)
            self.comp_h = nn.Linear(visual_channels, visual_channels, bias=False)
        else: 
            self.spatial_f = nn.Conv1d(visual_channels, hidden_size, 1, bias=False)
            self.spatial_g = nn.Conv1d(visual_channels, hidden_size, 1, bias=False)
            self.channel_f = nn.Linear(visual_flat, hidden_size, bias=False)
            self.channel_g = nn.Linear(visual_flat, hidden_size, bias=False)

        
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)

    def forward(self, visual_inputs):
        if self.attention_type == 'selfnew-sep-p':
            '''
                replace the matrix multiplication with point-wise multiplication,
                separate attention to similarity based channel-wise and spatial attention,
                get spatial and channel-wise attention mask from original feature map,
                the final attended feature map is the channel attended feature multiply by the spatial mask
            '''
            channel_f = self.channel_f(visual_inputs) # (batch_size, visual_channels, hidden_size)
            channel_g = self.channel_g(visual_inputs) # (batch_size, visual_channels, hidden_size)
            channel_sim = channel_f.matmul(channel_g.transpose(2, 1).contiguous()) # (batch_size, visual_channels, visual_channels)
            channel_sim_comp = channel_sim.sum(dim=1, keepdim=True) # (batch_size, 1, visual_channels)
            channel_mask = F.softmax(channel_sim_comp, dim=2).transpose(2, 1).contiguous() # (batch_size, visual_channels, 1)

            spatial_f = self.spatial_f(visual_inputs) # (batch_size, hidden_size, visual_flat)
            spatial_g = self.spatial_g(visual_inputs) # (batch_size, hidden_size, visual_flat)
            spatial_sim = spatial_f.transpose(2, 1).contiguous().matmul(spatial_g) # (batch_size, visual_flat, visual_flat)
            spatial_sim_comp = spatial_sim.sum(dim=1, keepdim=True) # (batch_size, 1, visual_flat)
            spatial_mask = F.softmax(spatial_sim_comp, dim=2) # (batch_size, 1, visual_flat)
            
            outputs = visual_inputs * channel_mask * spatial_mask # (batch_size, visual_channels, visual_flat)
            mask = (channel_mask, spatial_mask)
        elif self.attention_type == 'selfnew-sep-sf':
            '''
                replace the matrix multiplication with point-wise multiplication,
                separate attention to similarity based channel-wise and spatial attention
            '''
            spatial_f = self.spatial_f(visual_inputs) # (batch_size, hidden_size, visual_flat)
            spatial_g = self.spatial_g(visual_inputs) # (batch_size, hidden_size, visual_flat)
            spatial_sim = spatial_f.transpose(2, 1).contiguous().matmul(spatial_g) # (batch_size, visual_flat, visual_flat)
            spatial_sim_comp = spatial_sim.sum(dim=1, keepdim=True) # (batch_size, 1, visual_flat)
            spatial_mask = F.softmax(spatial_sim_comp, dim=2) # (batch_size, 1, visual_flat)
            feature = visual_inputs * spatial_mask # (batch_size, visual_channels, visual_flat)

            channel_f = self.channel_f(feature) # (batch_size, visual_channels, hidden_size)
            channel_g = self.channel_g(feature) # (batch_size, visual_channels, hidden_size)
            channel_sim = channel_f.matmul(channel_g.transpose(2, 1).contiguous()) # (batch_size, visual_channels, visual_channels)
            channel_sim_comp = channel_sim.sum(dim=1, keepdim=True) # (batch_size, 1, visual_channels)
            channel_mask = F.softmax(channel_sim_comp, dim=2).transpose(2, 1).contiguous() # (batch_size, visual_channels, 1)
            
            outputs = feature * channel_mask # (batch_size, visual_channels, visual_flat)
            mask = (channel_mask, spatial_mask)
        elif self.attention_type == 'selfnew-sep-cf':
            '''
                replace the matrix multiplication with point-wise multiplication,
                separate attention to similarity based channel-wise and spatial attention
            '''
            channel_f = self.channel_f(visual_inputs) # (batch_size, visual_channels, hidden_size)
            channel_g = self.channel_g(visual_inputs) # (batch_size, visual_channels, hidden_size)
            channel_sim = channel_f.matmul(channel_g.transpose(2, 1).contiguous()) # (batch_size, visual_channels, visual_channels)
            channel_sim_comp = channel_sim.sum(dim=1, keepdim=True) # (batch_size, 1, visual_channels)
            channel_mask = F.softmax(channel_sim_comp, dim=2).transpose(2, 1).contiguous() # (batch_size, visual_channels, 1)
            feature = visual_inputs * channel_mask # (batch_size, visual_channels, visual_flat)
            spatial_f = self.spatial_f(feature) # (batch_size, hidden_size, visual_flat)
            spatial_g = self.spatial_g(feature) # (batch_size, hidden_size, visual_flat)
            spatial_sim = spatial_f.transpose(2, 1).contiguous().matmul(spatial_g) # (batch_size, visual_flat, visual_flat)
            spatial_sim_comp = spatial_sim.sum(dim=1, keepdim=True) # (batch_size, 1, visual_flat)
            spatial_mask = F.softmax(spatial_sim_comp, dim=2) # (batch_size, 1, visual_flat)
            
            outputs = feature * spatial_mask # (batch_size, visual_channels, visual_flat)
            mask = (channel_mask, spatial_mask)
        elif self.attention_type == 'selfnew-nosep':
            '''
                separate attention to similarity based channel-wise and spatial attention
            '''
            feature = visual_inputs # (batch_size, visual_channels, visual_flat)
            channel_f = self.channel_f(feature) # (batch_size, visual_channels, hidden_size)
            channel_g = self.channel_g(feature) # (batch_size, visual_channels, hidden_size)
            channel_sim = channel_f.matmul(channel_g.transpose(2, 1).contiguous()) # (batch_size, visual_channels, visual_channels)
            channel_mask = F.softmax(channel_sim, dim=1).transpose(2, 1).contiguous() # (batch_size, visual_channels, visual_channels)
            feature = channel_mask.matmul(feature) # (batch_size, visual_channels, visual_flat)
            spatial_f = self.spatial_f(feature) # (batch_size, hidden_size, visual_flat)
            spatial_g = self.spatial_g(feature) # (batch_size, hidden_size, visual_flat)
            spatial_sim = spatial_f.transpose(2, 1).contiguous().matmul(spatial_g) # (batch_size, visual_flat, visual_flat)
            spatial_mask = F.softmax(spatial_sim, dim=1) # (batch_size, visual_flat, visual_flat)
            feature = feature.matmul(spatial_mask) # (batch_size, visual_channels, visual_flat)
            
            outputs = feature
            mask = (channel_mask, spatial_mask)
        elif self.attention_type == 'self-sep':
            '''
                self_sep
                replace the matrix multiplication with point-wise multiplication,
                and poisition-wise similarities over all positions for computing attention mask
            '''
            feature = visual_inputs.permute(0, 2, 1).contiguous() # (batch_size, visual_flat, visual_channels)
            f = self.comp_f(feature) # (batch_size, visual_flat, hidden_size)
            g = self.comp_g(feature) # (batch_size, visual_flat, hidden_size)
            h = self.comp_h(feature).transpose(2, 1).contiguous() # (batch_size, visual_channels, visual_flat)
            s = f.matmul(g.transpose(2, 1).contiguous()) # (batch_size, visual_flat, visual_flat)
            s_comp = torch.sum(s, dim=1, keepdim=True) # (batch_size, 1, visual_flat)
            mask = F.softmax(s_comp, dim=2) # (batch_size, 1, visual_flat)
            outputs = h * mask # (batch_size, visual_channels, visual_flat)
        elif self.attention_type == 'self-nosep':
            '''
                self_nosep
                vanilla self-attention module, use poisition-wise similarities for computing attention mask,
                see https://arxiv.org/pdf/1805.08318.pdf
            '''
            feature = visual_inputs.permute(0, 2, 1).contiguous() # (batch_size, visual_flat, visual_channels)
            f = self.comp_f(feature) # (batch_size, visual_flat, hidden_size)
            g = self.comp_g(feature) # (batch_size, visual_flat, hidden_size)
            h = self.comp_h(feature).transpose(2, 1).contiguous() # (batch_size, visual_channels, visual_flat)
            s = f.matmul(g.transpose(2, 1).contiguous()) # (batch_size, visual_flat, visual_flat)
            mask = F.softmax(s, dim=1) # (batch_size, visual_flat, visual_flat)
            outputs = h.matmul(mask) # (batch_size, visual_channels, visual_flat)
        else:
            raise ValueError("invalid attention type, terminating...")

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
        outputs = F.softmax(self.output_layer(h_comp), dim=1) # (batch_size, seq_size, 1)

        return outputs
