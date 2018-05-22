import math
import torch
import numpy as np
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

# attention module
class Attention2D(nn.Module):
    def __init__(self, visual_size, hidden_size, output_size):
        super(Attention2D, self).__init__()
        # basic settings
        self.visual_size = visual_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # parameters
        self.w_v = Parameter(torch.Tensor(visual_size, hidden_size))
        self.b_v = Parameter(torch.Tensor(output_size))
        self.w_h = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(torch.Tensor(output_size))
        self.w_o = Parameter(torch.Tensor(hidden_size, output_size))
        self.b_o = Parameter(torch.Tensor(output_size))
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, visual_inputs, states):
        # get the hidden state of the last LSTM layer
        # which is also the output of LSTM layer
        hidden = states[-1][0]
        # compute weighted sum of visual_inputs and hidden
        # visual_inputs = (batch_size, visual_size)
        # hidden = (batch_size, hidden_size)
        # outputs = (batch_size, hidden_size)

        # rescale
        hidden_min = hidden.min(1)[0].view(hidden.size(0), 1).expand_as(hidden)
        hidden_max = hidden.max(1)[0].view(hidden.size(0), 1).expand_as(hidden)
        hidden = (hidden - hidden_min) / (hidden_max - hidden_min)
        V = torch.matmul(visual_inputs, self.w_v) + self.b_v
        H = torch.matmul(hidden, self.w_h) + self.b_h
        # print(V[0].min(0)[0].item(), V[0].max(0)[0].item())
        # print(H[0].min(0)[0].item(), H[0].max(0)[0].item())
        # combine
        # outputs = (batch_size, output_size)
        outputs = F.tanh(V + H)
        outputs = torch.matmul(outputs, self.w_o) + self.b_o
        # compress to probability distribution
        outputs = F.softmax(outputs, dim=1)

        return outputs


# new LSTM with visual attention context
class AttentionLSTMCell2D(nn.Module):
    def __init__(self, visual_size, hidden_size):
        super(AttentionLSTMCell2D, self).__init__()
        # basic settings
        self.input_size = visual_size
        self.hidden_size = hidden_size
        # parameters
        for gate in ["i", "f", "c", "o"]:
            setattr(self, "w_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "u_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "z_{}".format(gate), Parameter(torch.Tensor(visual_size, hidden_size)))
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
    def __init__(self, batch_size, input_size, hidden_size, visual_channels, visual_size, num_layers=2, cuda_flag=True):
        super(AttentionDecoder2D, self).__init__()
        # basic settings
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.visual_channels = visual_channels
        self.visual_size = visual_size
        self.visual_flat = visual_size * visual_size
        self.visual_feature_size = visual_channels * visual_size * visual_size
        self.proj_size = 512
        self.feat_size = 512
        self.num_layers = num_layers
        self.cuda_flag = cuda_flag
        # layer settings
        # initialize hidden states
        self.init_h = nn.Linear(self.visual_channels, hidden_size)
        self.init_c = nn.Linear(self.visual_channels, hidden_size)
        # embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        # attention layer
        # in = (batch_size, 2 * hidden_size * num_layers)
        # out = (batch_size, visual_size * visual_size)
        
        # self.attention_layer = nn.Sequential(
        #     nn.Linear(self.feat_size + self.hidden_size, self.proj_size),
        #     nn.ReLU(),
        #     nn.Linear(self.proj_size, self.proj_size),
        #     nn.ReLU(),
        #     nn.Linear(self.proj_size, self.visual_flat),
        #     nn.Softmax()
        # )

        self.attention = Attention2D(self.visual_feature_size, self.hidden_size, self.visual_flat)
        # self.attention_layer = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.proj_size),
        #     nn.ReLU(),
        #     nn.Linear(self.proj_size, self.proj_size),
        #     nn.ReLU(),
        #     nn.Linear(self.proj_size, self.visual_flat),
        #     nn.Softmax()
        # )

        self.lstm_layer_1 = AttentionLSTMCell2D(self.visual_channels, self.hidden_size)
        self.lstm_layer_2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.visual_channels + self.hidden_size, self.proj_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.proj_size, self.input_size)
        )


    def init_hidden(self, visual_inputs):
        visual_flat = visual_inputs.view(visual_inputs.size(0), visual_inputs.size(1), visual_inputs.size(2) * visual_inputs.size(2))
        visual_flat = visual_flat.mean(2)
        states = [(
            self.init_h(visual_flat),
            self.init_c(visual_flat)
        ) for i in range(self.num_layers)]

        return states

    # def attend(self, visual_inputs, states):
    #     # compute attention weights
    #     # get the hidden state of the last LSTM layer
    #     # which is also the output of LSTM layer
    #     hidden = states[-1][0]
        
    #     # # concat visual_inputs and hidden
    #     # attention_inputs = torch.cat((visual_inputs, hidden), dim=1)
    #     # # use only hidden
    #     # attention_inputs = hidden
    #     # sum of visual_inputs and hidden (ensure the sizes are the same)
    #     attention_inputs = hidden + visual_inputs
    #     # # weighted sum of visual_inputs and hidden (ensure the sizes are the same)
    #     # alpha = 0.4
    #     # attention_inputs = hidden + alpha * visual_inputs

    #     # attention_inputs = (batch_size, hidden_size)
    #     # attention_weights = (batch_size, visual_size * visual_size)
    #     attention_weights = self.attention_layer(attention_inputs)

    #     return attention_weights

    def forward(self, visual_inputs, caption_inputs, states):
        # rescale
        batch_size = visual_inputs.size(0)
        visual_channels = visual_inputs.size(1)
        visual_size = visual_inputs.size(2)
        visual_inputs = visual_inputs.view(batch_size, visual_channels * visual_size * visual_size)
        visual_min = visual_inputs.min(1)[0].view(batch_size, 1).expand_as(visual_inputs)
        visual_max = visual_inputs.max(1)[0].view(batch_size, 1).expand_as(visual_inputs)
        visual_inputs = (visual_inputs - visual_min) / (visual_max - visual_min)
        visual_inputs = visual_inputs.view(batch_size, visual_channels, visual_size, visual_size)
        # feed
        seq_length = caption_inputs.size(1)
        batch_size = visual_inputs.size(0)
        decoder_outputs = []
        for step in range(seq_length):
            # embed words
            # caption_inputs = (batch_size)
            # embedded = (batch_size, hidden_size)
            embedded = self.embedding(caption_inputs[:, step])
            # get the attention weights
            # attention_weights = (batch_size, visual_size * visual_size)
            attention_weights = self.attention(visual_inputs.view(visual_inputs.size(0), self.visual_feature_size), states)
            # attention_weights = self.attend(visual_proj, states)
            # attended = (batch_size, visual_channels)
            attended = torch.matmul(
                visual_inputs.view(batch_size, self.visual_channels, self.visual_flat),
                attention_weights.view(batch_size, self.visual_flat, 1)    
            ).view(batch_size, self.visual_channels)
            # apply attention weights
            # feed into AttentionLSTM
            # outputs = (batch_size, hidden_size)
            states[0] = self.lstm_layer_1(embedded, states[0], attended)
            outputs = states[0][0]
            states[1] = self.lstm_layer_2(outputs, states[1])
            outputs = states[1][0]
            # get predicted probabilities
            # in = (batch_size, visual_channels + hidden_size)
            # out = (batch_size, 1, hidden_size)
            outputs = torch.cat((attended, outputs), dim=1)
            outputs = self.output_layer(outputs).unsqueeze(1)
            decoder_outputs.append(outputs)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs 

    def sample(self, visual_inputs, caption_inputs, states):
        # rescale
        batch_size = visual_inputs.size(0)
        visual_channels = visual_inputs.size(1)
        visual_size = visual_inputs.size(2)
        visual_inputs = visual_inputs.view(batch_size, visual_channels * visual_size * visual_size)
        visual_min = visual_inputs.min(1)[0].view(batch_size, 1).expand_as(visual_inputs)
        visual_max = visual_inputs.max(1)[0].view(batch_size, 1).expand_as(visual_inputs)
        visual_inputs = (visual_inputs - visual_min) / (visual_max - visual_min)
        visual_inputs = visual_inputs.view(batch_size, visual_channels, visual_size, visual_size)
        # feed
        # embed words
        # caption_inputs = (batch_size)
        # embedded = (batch_size, hidden_size)
        embedded = self.embedding(caption_inputs)
        # get the attention weights
        # attention_weights = (batch_size, visual_size * visual_size)
        attention_weights = self.attention(visual_inputs.view(visual_inputs.size(0), self.visual_feature_size), states)
        # attention_weights = self.attend(visual_proj, states)
        # attended = (batch_size, visual_channels)
        attended = torch.matmul(
            visual_inputs.view(batch_size, self.visual_channels, self.visual_flat),
            attention_weights.view(batch_size, self.visual_flat, 1)    
        ).view(batch_size, self.visual_channels)
        # apply attention weights
        # feed into AttentionLSTM
        # outputs = (batch_size, hidden_size)
        states[0] = self.lstm_layer_1(embedded, states[0], attended)
        outputs = states[0][0]
        states[1] = self.lstm_layer_2(outputs, states[1])
        outputs = states[1][0]
        # get predicted probabilities
        # in = (batch_size, visual_channels + hidden_size)
        # out = (batch_size, 1, hidden_size)
        outputs = torch.cat((attended, outputs), dim=1)
        outputs = self.output_layer(outputs).unsqueeze(1)

        return outputs, states, attention_weights

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
        # set mode
        self.encoder.eval()
        self.decoder.eval()

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

# for encoder-decoder pipeline with attention
class AttentionEncoderDecoder():
    def __init__(self, encoder_path, decoder_path, cuda_flag=True):
        if cuda_flag:
            self.encoder = torch.load(encoder_path).cuda()
            self.decoder = torch.load(decoder_path).cuda()
        else:
            self.encoder = torch.load(encoder_path)
            self.decoder = torch.load(decoder_path)
        # set mode
        self.encoder.eval()
        self.decoder.eval()

    def generate_text(self, image_inputs, dict_word2idx, dict_idx2word, max_length):
        caption_inputs = Variable(torch.LongTensor(np.reshape(np.array(dict_word2idx["<START>"]), (1)))).cuda()
        visual_contexts = self.encoder(image_inputs)
        # sample text indices via greedy search
        sampled = []
        states = self.decoder.init_hidden(visual_contexts)
        for i in range(max_length):
            outputs, states, _ = self.decoder.sample(visual_contexts, caption_inputs, states)
            # outputs = (1, 1, input_size)
            predicted = outputs.max(2)[1]
            # predicted = (1, 1)
            sampled.append(predicted)
            caption_inputs = predicted.view(1)
            if dict_idx2word[caption_inputs[-1].view(1).cpu().numpy()[0]] == '<END>':
                break
        sampled = torch.cat(sampled)
        # decoder indices to words
        captions = []
        for sequence in sampled.cpu().numpy():
            caption = []
            for index in sequence:
                word = dict_idx2word[index]
                caption.append(word)
                if word == '<END>':
                    break
            captions.append(" ".join(caption))

        return captions

    # image_inputs = (1, visual_channels, visual_size, visual_size)
    # caption_inputs = (1)
    def visual_attention(self, image_inputs, dict_word2idx, dict_idx2word, max_length):
        caption_inputs = Variable(torch.LongTensor(np.reshape(np.array(dict_word2idx["<START>"]), (1)))).cuda()
        visual_contexts = self.encoder(image_inputs)
        # sample text indices via greedy search
        pairs = []
        states = self.decoder.init_hidden(visual_contexts)
        for i in range(max_length):
            outputs, states, attention_weights = self.decoder.sample(visual_contexts, caption_inputs, states)
            # attentions = (visual_size, visual_size)
            predicted = outputs.max(2)[1]
            # predicted = (1, 1)
            caption_inputs = predicted.view(1)
            word = dict_idx2word[predicted.cpu().numpy()[0][0]]
            attention_weights = F.upsample_bilinear(attention_weights.view(1, 1, visual_contexts.size(2), visual_contexts.size(2)), size=(64, 64))
            pairs.append((word, attention_weights.view(64, 64)))
            if word == '<END>':
                break

        return pairs