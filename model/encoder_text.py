import torch
import torch.nn as nn

class ShapenetTextEncoder(nn.Module):
    def __init__(self, dict_size):
        super(ShapenetTextEncoder, self).__init__()
        # embedding
        self.embedding = nn.Embedding(dict_size, 128)

        # first conv block
        self.conv_128 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.bn_128 = nn.BatchNorm2d(128)

        # second conv block
        self.conv_256 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.bn_256 = nn.BatchNorm2d(256)

        # recurrent block
        self.init_h = nn.Linear(256, 512)
        self.lstm = nn.LSTM(256, 512, batch_first=True)

        # output block
        self.outputs = nn.Sequential(
            nn.Linear(512, 256),
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
        h = self.init_h(conved.mean(1)).unsqueeze(0)
        if h.is_cuda:
            c = torch.zeros(1, h.size(1), 512).cuda()
        else:
            c = torch.zeros(1, h.size(1), 512)
        _, (encoded, _) = self.lstm(conved, (h, c))

        #################
        # outputs
        #################
        outputs = self.outputs(encoded.squeeze())

        return outputs
