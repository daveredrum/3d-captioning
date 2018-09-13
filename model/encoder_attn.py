import torch
import torch.nn as nn
import torch.nn.functional as F
from model.net_components import AdaptiveLSTMCell
from model.attn_components import *

class SelfAttnShapeEncoder(nn.Module):
    def __init__(self):
        super(SelfAttnShapeEncoder, self).__init__()
        self.shape_conv_1 = nn.Sequential(
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
        self.attention_spatial_1 = SelfAttention3D(256, 256, 512)
        self.shape_conv_2 = nn.Sequential(
            nn.Conv3d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(512)
        )
        self.attention_spatial_2 = SelfAttention3D(512, 256, 64)
        self.shape_outputs = nn.Linear(512, 128)
        # self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = nn.Parameter(torch.ones(1))
    
    def _get_shape_feat(self, inputs, conv_layer, flat=True):
        conved = conv_layer(inputs)
        if flat:
            return conved.view(inputs.size(0), conved.size(1), -1).contiguous()
        else:
            return conved

    def attend(self, shape_feat, attn_layer):
        spatial_contexts, spatial_weights = attn_layer(shape_feat)
        # spatial_attended = shape_feat + self.gamma * spatial_contexts # (batch_size, visual_channels, visual_flat)
        # torch.clamp(self.gamma, 0., 1.)
        spatial_attended = shape_feat + spatial_contexts # (batch_size, visual_channels, visual_flat)

        return spatial_attended, spatial_weights

    def forward(self, shape_inputs):
        # get features
        shape_feat_1 = self._get_shape_feat(shape_inputs, self.shape_conv_1) # (batch_size, 256, 512)
        spatial_attended_1, spatial_weights_1 = self.attend(shape_feat_1, self.attention_spatial_1)
        spatial_attended_1 = spatial_attended_1.view(shape_inputs.size(0), 256, 8, 8, 8) # (batch_size, 256, 8, 8, 8)
        shape_feat_2 = self._get_shape_feat(spatial_attended_1, self.shape_conv_2) # (batch_size, 512, 64)
        spatial_attended_2, spatial_weights_2 = self.attend(shape_feat_2, self.attention_spatial_2)
        # outputs
        shape_outputs = self.shape_outputs(spatial_attended_2.mean(2)) # (batch_size, 128)
        spatial_weights = (spatial_weights_1, spatial_weights_2)

        return shape_outputs, spatial_weights


class SelfAttnTextEncoder(nn.Module):
    def __init__(self, dict_size):
        super(SelfAttnTextEncoder, self).__init__()
        self.text_embedding = nn.Embedding(dict_size, 128, padding_idx=0)
        self.text_conv_128 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.text_bn_128 = nn.BatchNorm2d(128)
        self.text_conv_256 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1, bias=False),
            nn.ReLU()
        )
        self.text_bn_256 = nn.BatchNorm2d(256)
        self.text_outputs = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.lstm_cell = nn.LSTM(256, 256, batch_first=True) # more efficient than the LSTMCell
        self.attention_temporal = TemporalSelfAttention(256)
    
    def _get_text_feat(self, inputs):
        embedded = self.text_embedding(inputs) # (batch_size, seq_size, 128)
        conved = self.text_conv_128(embedded.transpose(2, 1).contiguous()) # (batch_size, 128, seq_size)
        conved = self.text_bn_128(conved.unsqueeze(3))
        conved = conved.view(*list(conved.size())[:-1]) # (batch_size, 128, seq_size)
        conved = self.text_conv_256(conved) # (batch_size, 256, seq_size)
        conved = self.text_bn_256(conved.unsqueeze(3))
        conved = conved.view(*list(conved.size())[:-1]).transpose(2, 1).contiguous() # (batch_size, seq_size, 256)

        return conved
    
    def _init_hidden(self, text_feat):
        states = (
            torch.zeros(1, text_feat.size(0), 256).cuda(),
            torch.zeros(1, text_feat.size(0), 256).cuda()
        )

        return states

    def attend(self, text_feat, states):
        temporal_weights = self.attention_temporal(text_feat)
        temporal_attended = torch.sum(text_feat * temporal_weights, dim=1) # (batch_size, hidden_size)

        return temporal_attended, temporal_weights

    def forward(self, text_inputs):
        text_feat = self._get_text_feat(text_inputs) # (batch_size, seq_size, 256)
        states = self._init_hidden(text_feat) # (batch_size, 256)
        text_feat, _ = self.lstm_cell(text_feat, states) # (batch_size, seq_size, 256)
        temporal_attended, temporal_weights = self.attend(text_feat, states)
        # outputs
        text_outputs = self.text_outputs(temporal_attended)

        return text_outputs, temporal_weights
