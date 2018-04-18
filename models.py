import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 58 * 58, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
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


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=True):
        super(Decoder, self).__init__()
        # the size of inputs and outputs should be equal to the size of the dictionary
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm_layer = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.cuda_flag = cuda_flag

    def forward(self, visual_inputs, caption_inputs, length_list):
        embedded = self.embedding(caption_inputs)
        # initialize h and c as zeros
        if self.cuda_flag:
            hiddens = (
                Variable(torch.zeros(*(list(visual_inputs.size())))).cuda(), 
                Variable(torch.zeros(*(list(visual_inputs.size())))).cuda()
                )
        else:
            hiddens = (
                Variable(torch.zeros(*(list(visual_inputs.size())))), 
                Variable(torch.zeros(*(list(visual_inputs.size()))))
                )
        # concatenate the visual input with embedded vectors
        embedded = torch.cat((visual_inputs.unsqueeze(1), embedded), 1)
        # pack captions of different length
        packed = pack_padded_sequence(embedded, length_list, batch_first=True)
        outputs, _ = self.lstm_layer(embedded, hiddens)
        outputs = self.output_layer(outputs)

        return outputs