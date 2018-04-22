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
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
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

    # chop the last output layer
    def extract(self, inputs):
        outputs = self.conv_layer(inputs).view(inputs.size(0), -1)
        outputs = self.fc_layer(outputs)
        
        return outputs


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, cuda_flag=True):
        super(Decoder, self).__init__()
        # the size of inputs and outputs should be equal to the size of the dictionary
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm_layer = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            # omitted softmax layer if using cross entropy loss
            nn.LogSoftmax() # if using NLLLoss (softmax layer + NLLLoss = CrossEntropyLoss)
        )
        self.cuda_flag = cuda_flag

    def forward(self, visual_inputs, caption_inputs, length_list):
        embedded = self.embedding(caption_inputs)
        # concatenate the visual input with embedded vectors
        embedded = torch.cat((visual_inputs.unsqueeze(1), embedded), 1)
        # pack captions of different length
        packed = pack_padded_sequence(embedded, length_list, batch_first=True)
        # hiddens = (outputs, states)
        hiddens, _ = self.lstm_layer(packed, None)
        outputs = self.output_layer(hiddens[0])

        return outputs, hiddens[1]

# pipeline for pretrained encoder-decoder pipeline
class Pipeline():
    def __init__(self, encoder_path, decoder_path, cuda_flag=True):
        if cuda_flag:
            self.encoder = torch.load(encoder_path).cuda()
            self.decoder = torch.load(decoder_path).cuda()
        else:
            self.encoder = torch.load(encoder_path)
            self.decoder = torch.load(decoder_path)

    def generate_text(self, image_inputs, dictionary, max_length):
        inputs = self.encoder.extract(image_inputs).unsqueeze(1)
        states = None
        # sample text indices via greedy search
        sampled = []
        for i in range(max_length):
            outputs, states = self.decoder.lstm_layer(inputs, states)
            outputs = self.decoder.output_layer(outputs[0])
            predicted = outputs.max(1)[1]
            sampled.append(predicted.view(-1, 1))
            inputs = self.decoder.embedding(predicted).unsqueeze(1)
        sampled = torch.cat(sampled, 1)
        # decoder indices to words
        captions = []
        for sequence in sampled.cpu().numpy():
            caption = []
            for index in sequence:
                word = dictionary[index]
                caption.append(word)
                if word == '<END>':
                    break
            captions.append(" ".join(caption))

        return captions

    