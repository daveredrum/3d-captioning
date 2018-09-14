import torch
import torch.nn as nn

class ShapenetTextEncoder(nn.Module):
    def __init__(self, dict_size):
        super(ShapenetTextEncoder, self).__init__()
        # embedding
        self.embedding = nn.Embedding(dict_size, 128)
        self.conv_128 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.bn_128 = nn.BatchNorm2d(128)
        self.conv_256 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1, bias=False),
            nn.ReLU()
        )
        self.bn_256 = nn.BatchNorm2d(256)

        # recurrent block
        self.lstm = nn.LSTM(256, 256, batch_first=True)

        # output block
        self.outputs = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, inputs):
        #################
        # convolutional
        #################
        embedded = self.embedding(inputs) # (batch_size, seq_size, emb_size)
        conved = self.conv_128(embedded.transpose(2, 1).contiguous()) # (batch_size, emb_size, seq_size)
        conved = self.bn_128(conved.unsqueeze(3)).squeeze() # (batch_size, emb_size, seq_size)
        conved = self.conv_256(conved) # (batch_size, emb_size, seq_size)
        conved = self.bn_256(conved.unsqueeze(3)).squeeze().transpose(2, 1).contiguous() # (batch_size, seq_size, emb_size)

        #################
        # recurrent
        ################# 
        encoded, _ = self.lstm(conved, None)

        #################
        # outputs
        #################
        outputs = self.outputs(encoded[:, -1, :].squeeze())

        return outputs
