import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as torchmodels
import torch.nn.functional as F
from torch.nn import Parameter

# decoder without attention
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
            # nn.LogSoftmax() # if using NLLLoss (softmax layer + NLLLoss = CrossEntropyLoss)
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

    def sample(self, visual_inputs, length_list):
        batch_size = visual_inputs.size(0)
        states = None
        # sample text indices via greedy search
        sampled = []
        for batch in range(batch_size):
            inputs = visual_inputs[batch].view(1, 1, -1)
            for i in range(length_list[batch]):
                outputs, states = self.lstm_layer(inputs, states)
                outputs = self.output_layer(outputs)
                predicted = outputs.max(2)[1]
                sampled.append(outputs.view(1, -1))
                inputs = self.embedding(predicted)
        sampled = torch.cat(sampled, 0)

        return sampled

# attention module for image encoder
# implement soft attention
class Attention2D(nn.Module):
    def __init__(self, visual_channels, visual_size, hidden_size, num_layers, cuda_flag=True):
        super(Attention2D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.visual_size = visual_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.visual_flat_size = visual_channels * visual_size * visual_size
        # layers
        self.attention = nn.Linear(self.visual_flat_size + hidden_size * num_layers, self.visual_flat_size)
        self.attention_out = nn.Linear(visual_channels, hidden_size)

    # inputs:   
    # visual_inputs = (batch, visual_channels, visual_size, visual_size)
    # states_h = (batch, hidden_size)
    # outputs:
    # attended = (batch, hidden_size)
    def forward(self, visual_inputs, states_h):
        # settings
        batch_size = visual_inputs.size(0)
        visual_inputs = visual_inputs.view(batch_size, -1)
        # compute attention weights
        attention_inputs = torch.cat((visual_inputs, states_h), dim=1)
        attention_weights = self.attention(attention_inputs)
        attention_weights = F.softmax(attention_weights, dim=1)
        attention_weights = attention_weights.view(batch_size, self.visual_channels, self.visual_size, self.visual_size)
        print(attention_weights.size())
        print(visual_inputs.size())
        # apply attention weights
        attention_applied = visual_inputs * attention_weights
        attention_applied = attention_applied.view(batch_size, self.visual_channels, self.visual_size * self.visual_size)
        attention_applied = torch.sum(attention_applied, dim=2)
        # outputs
        attended = self.attention_out(attention_applied)

        return attended

# new LSTM with visual attention context
class AttentionLSTMCell2D(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTMCell2D, self).__init__()
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
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    # inputs = (batch, input_size)
    # states_h = (batch, hidden_size)
    # states_c = (batch, hidden_size)
    # atteded = (batch, hidden_size)
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


# decoder with attention
class AttentionDecoder2D(nn.Module):
    def __init__(self, input_size, hidden_size, visual_channels, visual_size, num_layers=2, cuda_flag=True):
        super(AttentionDecoder2D, self).__init__()
        # basic settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.visual_channels = visual_channels
        self.visual_size = visual_size
        self.num_layers = num_layers
        self.cuda_flag = cuda_flag
        # layer settings
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attention = Attention2D(visual_channels, visual_size, hidden_size, num_layers)
        self.lstm_layer = [AttentionLSTMCell2D(input_size, hidden_size)] + [nn.LSTMCell(hidden_size, hidden_size) for i in range(num_layers - 1)]
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, visual_inputs, caption_inputs):
        seq_length = caption_inputs.size(1)
        batch_size = visual_inputs.size(0)
        if self.cuda_flag:
            states = [(
                torch.zeros(batch_size, self.hidden_size).cuda(),
                torch.zeros(batch_size, self.hidden_size).cuda()
            ) for i in range(self.num_layers)]
        else:
            states = [(
                torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size)
            ) for i in range(self.num_layers)]
        decoder_outputs = []
        for step in range(seq_length):
            # get the attention weights
            # hidden = (batch_size, hidden_size * num_layers)
            hidden = torch.cat([states[i][0] for i in range(self.num_layers)], dim=1)
            # attended = (batch_size, hidden_size)
            attended = self.attention(visual_inputs, hidden)
            # embed words
            # caption_inputs = (batch_size)
            # embedded = (batch_size, hidden_size)
            embedded = self.embedding(caption_inputs[:, step])
            # apply attention weights
            # feed into AttentionLSTM
            # outputs = (batch_size, hidden_size)
            for layer_id in range(self.num_layers):
                if layer_id == 0:
                    states[layer_id] = self.lstm_layer[layer_id](embedded, states[layer_id], attended)
                    outputs = states[layer_id][0]
                else:
                    states[layer_id] = self.lstm_layer[layer_id](outputs, states[layer_id])
                    outputs = states[layer_id][0]
            # get predicted probabilities
            # outputs = (batch_size, 1, hidden_size)
            outputs = self.output_layer(outputs).unsqueeze(1)
            decoder_outputs.append(outputs)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs 

# pipeline for pretrained encoder-decoder pipeline
# same pipeline for both 2d and 3d
class EncoderDecoder():
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