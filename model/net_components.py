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


# new LSTM with visual attention context
class Att2AllLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Att2AllLSTMCell, self).__init__()
        # basic settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        # parameters
        for gate in ["i", "f", "c", "o"]:
            setattr(self, "w_{}".format(gate), Parameter(torch.Tensor(input_size, hidden_size)))
            setattr(self, "u_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "z_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "b_{}".format(gate), Parameter(torch.Tensor(hidden_size)))
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    # inputs = (batch, input_size)
    # states_h = (batch, hidden_size)
    # states_c = (batch, hidden_size)
    # atteded = (batch, input_size)
    # outputs = (states_h, states_c)
    def forward(self, embedded, states, atteded):
        # unpack states
        states_h, states_c = states
        # forward feed
        i = F.sigmoid(torch.matmul(embedded, self.w_i) + torch.matmul(states_h, self.u_i) + torch.matmul(atteded, self.z_i) + self.b_i)
        f = F.sigmoid(torch.matmul(embedded, self.w_f) + torch.matmul(states_h, self.u_f) + torch.matmul(atteded, self.z_f) + self.b_f)
        c_hat = F.tanh(torch.matmul(embedded, self.w_c) + torch.matmul(states_h, self.u_c) + torch.matmul(atteded, self.z_c) + self.b_c)
        states_c = f * states_c + i * c_hat
        o = F.sigmoid(torch.matmul(embedded, self.w_o) + torch.matmul(states_h, self.u_o) + torch.matmul(atteded, self.z_o) + self.b_o)
        states_h = o * F.tanh(states_c)
        # pack states
        states = (states_h, states_c)

        return states

# new LSTM with visual attention context
class Att2InLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Att2InLSTMCell, self).__init__()
        # basic settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        # parameters
        for gate in ["i", "f", "c", "o"]:
            setattr(self, "w_{}".format(gate), Parameter(torch.Tensor(input_size, hidden_size)))
            setattr(self, "u_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "b_{}".format(gate), Parameter(torch.Tensor(hidden_size)))
        setattr(self, "z_c", Parameter(torch.Tensor(hidden_size, hidden_size)))
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    # inputs = (batch, input_size)
    # states_h = (batch, hidden_size)
    # states_c = (batch, hidden_size)
    # atteded = (batch, input_size)
    # outputs = (states_h, states_c)
    def forward(self, embedded, states, atteded):
        # unpack states
        states_h, states_c = states
        # forward feed
        i = F.sigmoid(torch.matmul(embedded, self.w_i) + torch.matmul(states_h, self.u_i) + self.b_i)
        f = F.sigmoid(torch.matmul(embedded, self.w_f) + torch.matmul(states_h, self.u_f) + self.b_f)
        c_hat = F.tanh(torch.matmul(embedded, self.w_c) + torch.matmul(states_h, self.u_c) + torch.matmul(atteded, self.z_c) + self.b_c)
        states_c = f * states_c + i * c_hat
        o = F.sigmoid(torch.matmul(embedded, self.w_o) + torch.matmul(states_h, self.u_o) + self.b_o)
        states_h = o * F.tanh(states_c)
        # pack states
        states = (states_h, states_c)

        return states

# LSTM with sentinel gate
class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        # basic settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        # parameters
        for gate in ["i", "f", "c", "o", 's']:
            setattr(self, "w_{}".format(gate), Parameter(torch.Tensor(input_size, hidden_size)))
            setattr(self, "u_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "b_{}".format(gate), Parameter(torch.Tensor(hidden_size)))
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    # inputs = (batch, input_size)
    # states_h = (batch, hidden_size)
    # states_c = (batch, hidden_size)
    # states = (states_h, states_c)
    # sentinel = (batch, hidden_size)
    def forward(self, embedded, states):
        # unpack states
        states_h, states_c = states
        # forward feed
        i = F.sigmoid(torch.matmul(embedded, self.w_i) + torch.matmul(states_h, self.u_i) + self.b_i)
        f = F.sigmoid(torch.matmul(embedded, self.w_f) + torch.matmul(states_h, self.u_f) + self.b_f)
        c_hat = F.tanh(torch.matmul(embedded, self.w_c) + torch.matmul(states_h, self.u_c) + self.b_c)
        states_c = f * states_c + i * c_hat
        o = F.sigmoid(torch.matmul(embedded, self.w_o) + torch.matmul(states_h, self.u_o) + self.b_o)
        states_h = o * F.tanh(states_c)
        # pack states
        states = (states_h, states_c)
        # sentinel gate
        s_hat = F.sigmoid(torch.matmul(embedded, self.w_s) + torch.matmul(states_h, self.u_s) + self.b_s)
        sentinel = s_hat * F.tanh(states_c)

        return states, sentinel