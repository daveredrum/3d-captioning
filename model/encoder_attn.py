import torch
import torch.nn as nn
import torch.nn.functional as F
from model.net_components import AdaptiveLSTMCell
from model.attn_components import *

class AdaptiveEncoder(nn.Module):
    def __init__(self, dict_size, ver):
        super(AdaptiveEncoder, self).__init__()
        self.ver = ver
        ############################
        #                          #
        #       shape encoder      #
        #                          #
        ############################
        self.shape_conv = nn.Sequential(
            nn.Conv3d(4, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(256)
        )
        self.shape_outputs = nn.Linear(256, 128)

        ###########################
        #                         #
        #       text encoder      #
        #                         #
        ###########################
        self.text_embedding = nn.Embedding(dict_size, 128)
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

        ###########################
        #                         #
        #         attention       #
        #                         #
        ###########################
        if self.ver == "1" or self.ver == "2":
            self.lstm_cell = AdaptiveLSTMCell(256, 256)
            self.attention = AdaptiveAttention3D(256, 256, 512)
        elif self.ver == "2.1-a" or self.ver == "2.1-b":
            self.lstm_cell = nn.LSTMCell(256, 256)
            self.attention_spatial = Attention3D(256, 256, 512)
            self.attention_temporal = TemporalAttention(256, self.ver)
        elif self.ver == "2.1-c":
            self.lstm_cell = AdaptiveLSTMCell(256, 256)
            self.attention_spatial = AdaptiveSpatialAttention(256, 256, 512)
            self.attention_temporal = AdaptiveTemporalAttention(256)
        else:
            raise ValueError("invalid version, terminating...")

    def _get_shape_feat(self, inputs, flat=True):
        conved = self.shape_conv(inputs)
        if flat:
            return conved.view(inputs.size(0), conved.size(1), -1).contiguous()
        else:
            return conved

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
        return (
            torch.zeros(text_feat.size(0), 256).cuda(),
            torch.zeros(text_feat.size(0), 256).cuda()
        )

    def attend(self, shape_feat, text_feat, states):
        if self.ver == "1":
            states, sentinel = self.lstm_cell(text_feat, states)
            h, c = states
            weights_step = self.attention(shape_feat, states, sentinel)
            attention_weights = weights_step[:, :-1]
            sentinel_scalar = weights_step[:, -1].unsqueeze(1)
            attended = torch.sum(shape_feat * attention_weights.unsqueeze(1), 2)
            shape_contexts_step = (1 - sentinel_scalar) * attended
            text_contexts_step = sentinel_scalar * sentinel + shape_contexts_step
            states = (text_contexts_step + h, c)
        elif self.ver == "2":
            states, sentinel = self.lstm_cell(text_feat, states)
            weights_step = self.attention(shape_feat, states, sentinel)
            attention_weights = weights_step[:, :-1]
            attended = torch.sum(shape_feat * attention_weights.unsqueeze(1), 2)
            shape_contexts_step = attended
            text_contexts_step = states[0]
        elif self.ver == "2.1-a" or self.ver == "2.1-b":
            states = self.lstm_cell(text_feat, states)
            weights_step = self.attention_spatial(shape_feat, states)
            shape_contexts_step = torch.sum(shape_feat * weights_step.unsqueeze(1), 2)
            text_contexts_step = states[0]
        elif self.ver == "2.1-c":
            states, sentinel = self.lstm_cell(text_feat, states)
            weights_step, sentinel_scalar_step = self.attention_spatial(shape_feat, states, sentinel)
            shape_contexts_step = torch.sum(shape_feat * weights_step.unsqueeze(1), 2)
            text_contexts_step = states[0]
            weights_step = torch.cat((weights_step, sentinel_scalar_step), dim=1)
        else:
            raise ValueError("invalid version, terminating...")

        return shape_contexts_step, text_contexts_step, states, weights_step


    def forward(self, shape_inputs, text_inputs):
        # get features
        shape_feat = self._get_shape_feat(shape_inputs) # (batch_size, 256, 512)
        text_feat = self._get_text_feat(text_inputs) # (batch_size, seq_size, 256)
        states = self._init_hidden(text_feat) # (batch_size, 256)
        
        # through attention
        if self.ver == "1":
            shape_contexts = []
            weights = []
            for i in range(text_feat.size(1)):
                shape_contexts_step, states, weights_step = self.attend(shape_feat, text_feat[:, i, :], states)
                shape_contexts.append(shape_contexts_step.unsqueeze(2))
                weights.append(weights_step)
            
            shape_attended = torch.cat(shape_contexts, dim=2).mean(2)
            text_attended = states[0]
        elif self.ver == "2":
            shape_contexts = []
            text_contexts = []
            weights = []
            sentinel_scalars = []
            for i in range(text_feat.size(1)):
                shape_contexts_step, text_contexts_step, states, weights_step = self.attend(shape_feat, text_feat[:, i, :], states)
                shape_contexts.append(shape_contexts_step.unsqueeze(2))
                text_contexts.append(text_contexts_step.unsqueeze(2))
                weights.append(weights_step)
                sentinel_scalars.append(weights_step[:, -1].unsqueeze(1))
            
            attn_mask = F.softmax(1. - torch.cat(sentinel_scalars, dim=1), dim=1).unsqueeze(1)
            shape_attended = torch.sum(torch.cat(shape_contexts, dim=2) * attn_mask, dim=2)
            text_attended = torch.sum(torch.cat(text_contexts, dim=2) * attn_mask, dim=2)
        elif self.ver == "2.1-a" or self.ver == "2.1-b":
            shape_contexts = []
            text_contexts = []
            weights = []
            for i in range(text_feat.size(1)):
                shape_contexts_step, text_contexts_step, states, weights_step = self.attend(shape_feat, text_feat[:, i, :], states)
                shape_contexts.append(shape_contexts_step.unsqueeze(2))
                text_contexts.append(text_contexts_step.unsqueeze(2))
                weights.append(weights_step)

            # temporal attention
            shape_contexts = torch.cat(shape_contexts, dim=2)
            text_contexts = torch.cat(text_contexts, dim=2)
            attn_mask = self.attention_temporal(shape_contexts, text_contexts)
            shape_attended = torch.sum(shape_contexts * attn_mask, dim=2)
            text_attended = torch.sum(text_contexts * attn_mask, dim=2)
        elif self.ver == "2.1-c":
            shape_contexts = []
            text_contexts = []
            sentinel_scalars = []
            weights = []
            for i in range(text_feat.size(1)):
                shape_contexts_step, text_contexts_step, states, weights_step = self.attend(shape_feat, text_feat[:, i, :], states)
                shape_contexts.append(shape_contexts_step.unsqueeze(2))
                text_contexts.append(text_contexts_step.unsqueeze(2))
                sentinel_scalars.append(weights_step[:, -1].view(weights_step.size(0), 1, 1))
                weights.append(weights_step)

            # temporal attention
            shape_contexts = torch.cat(shape_contexts, dim=2)
            text_contexts = torch.cat(text_contexts, dim=2)
            sentinel_scalars = torch.cat(sentinel_scalars, dim=2)
            attn_mask = self.attention_temporal(shape_contexts, text_contexts, sentinel_scalars)
            shape_attended = torch.sum(shape_contexts * attn_mask, dim=2)
            text_attended = torch.sum(text_contexts * attn_mask, dim=2)
        else:
            raise ValueError("invalid version, terminating...")

        # outputs
        shape_outputs = self.shape_outputs(shape_attended)
        text_outputs = self.text_outputs(text_attended)

        return shape_outputs, text_outputs, weights, attn_mask
        
