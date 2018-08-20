import torch
import torch.nn as nn
from model.net_components import AdaptiveLSTMCell
from model.attn_components import AdaptiveAttention3D

class AdaptiveEncoder(nn.Module):
    def __init__(self, dict_size):
        super(AdaptiveEncoder, self).__init__()
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
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
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
        self.lstm_cell = AdaptiveLSTMCell(256, 256)
        self.attention = AdaptiveAttention3D(256, 256, 512)

    def _get_shape_feat(self, inputs, flat=True):
        conved = self.shape_conv(inputs)
        if flat:
            return conved.view(inputs.size(0), conved.size(1), -1).contiguous()
        else:
            return conved

    def _get_text_feat(self, inputs):
        embedded = self.text_embedding(inputs) # (batch_size, seq_size, 128)
        conved = self.text_conv_128(embedded.transpose(2, 1).contiguous()) # (batch_size, 128, seq_size)
        conved = self.text_bn_128(conved.unsqueeze(3)).squeeze() # (batch_size, 128, seq_size)
        conved = self.text_conv_256(conved) # (batch_size, 256, seq_size)
        conved = self.text_bn_256(conved.unsqueeze(3)).squeeze().transpose(2, 1).contiguous() # (batch_size, seq_size, 256)

        return conved

    def _init_hidden(self, text_feat):
        return (
            torch.zeros(text_feat.size(0), 256).cuda(),
            torch.zeros(text_feat.size(0), 256).cuda()
        )

    def attend(self, shape_feat, text_feat, states):
        states, sentinel = self.lstm_cell(text_feat, states)
        h, c = states
        weights = self.attention(shape_feat, states, sentinel)
        attention_weights = weights[:, :-1]
        sentinel_scalar = weights[:, -1].unsqueeze(1)
        attended = torch.sum(shape_feat * attention_weights.unsqueeze(1), 2)
        shape_contexts_step = (1 - sentinel_scalar) * attended
        attended = sentinel_scalar * sentinel + shape_contexts_step
        states = (attended + h, c)

        return shape_contexts_step, states, weights


    def forward(self, shape_inputs, text_inputs):
        # get features
        shape_feat = self._get_shape_feat(shape_inputs) # (batch_size, 256, 512)
        text_feat = self._get_text_feat(text_inputs) # (batch_size, seq_size, 256)
        states = self._init_hidden(text_feat) # (batch_size, 256)
        
        # through attention
        shape_contexts = []
        for i in range(text_feat.size(1)):
            shape_contexts_step, states, _ = self.attend(shape_feat, text_feat[:, i, :], states)
            shape_contexts.append(shape_contexts_step.unsqueeze(2))
        shape_attended = torch.cat(shape_contexts, dim=2).mean(2)
        text_attended = states[0]

        # outputs
        shape_outputs = self.shape_outputs(shape_attended)
        text_outputs = self.text_outputs(text_attended)

        return shape_outputs, text_outputs
        
