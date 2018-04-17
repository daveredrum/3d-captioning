import torch
import torch.nn as nn
from torch.autograd import Variable

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
    def __init__(self, input_size, hidden_size):
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

    def forward(self, inputs, hiddens):
        embedded = self.embedding(inputs).view(1, 1, -1)
        outputs, hiddens = self.lstm_layer(embedded, hiddens)
        outputs = self.output_layer(outputs)

        return outputs, hiddens