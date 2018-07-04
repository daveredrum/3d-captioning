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


# decoder without attention
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        # the size of inputs and outputs should be equal to the size of the dictionary
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm_layer = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            # nn.Dropout(p=0.2)
        )

    def init_hidden(self, visual_inputs):
        states = (
            Variable(torch.zeros(visual_inputs.size(0), self.hidden_size), requires_grad=False).cuda(),
            Variable(torch.zeros(visual_inputs.size(0), self.hidden_size), requires_grad=False).cuda()
        )

        return states

    def forward(self, features, caption_inputs, states):
        # feed
        seq_length = caption_inputs.size(1) + 1
        decoder_outputs = []
        for step in range(seq_length):
            if step == 0:
                embedded = features
            else:
                embedded = self.embedding(caption_inputs[:, step - 1])
            states = self.lstm_layer(embedded, states)
            lstm_outputs = states[0]
            outputs = self.output_layer(lstm_outputs).unsqueeze(1)
            decoder_outputs.append(outputs)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs


    def sample(self, embedded, states):
        new_states = self.lstm_layer(embedded, states)
        lstm_outputs = new_states[0]
        outputs = self.output_layer(lstm_outputs).unsqueeze(1)

        return outputs, new_states

    def beam_search(self, features, beam_size, max_length):
        batch_size = features.size(0)
        outputs = []
        for feat_id in range(batch_size):
            feature = features[feat_id].unsqueeze(0)
            states = self.init_hidden(feature)
            start, states = self.sample(feature, states)
            start = F.log_softmax(start, dim=2)
            start_scores, start_words = start.topk(beam_size, dim=2)[0].squeeze(), start.topk(beam_size, dim=2)[1].squeeze()
            # a queue containing all searched words and their log_prob
            searched = deque([([start_words[i].view(1)], start_scores[i].view(1), states) for i in range(beam_size)])
            done = []
            while True:
                candidate = searched.popleft()
                prev_word, prev_prob, prev_states = candidate
                if len(prev_word) <= max_length and int(prev_word[-1].item()) != 3:
                    embedded = self.embedding(prev_word[-1])
                    preds, new_states = self.sample(embedded, prev_states)
                    preds = F.log_softmax(preds, dim=2)
                    top_scores, top_words = preds.topk(beam_size, dim=2)[0].squeeze(), preds.topk(beam_size, dim=2)[1].squeeze()
                    for i in range(beam_size):
                        next_word, next_prob = copy.deepcopy(prev_word), prev_prob.clone()
                        next_word.append(top_words[i].view(1))
                        next_prob += top_scores[i].view(1)
                        searched.append((next_word, next_prob, new_states))
                    searched = deque(sorted(searched, reverse=True, key=lambda s: s[1])[:beam_size])    
                else:
                    done.append((prev_word, prev_prob))
                if not searched:
                    break       
            
            done = sorted(done, reverse=True, key=lambda s: s[1])
            best = [word[0].item() for word in done[0][0]]
            outputs.append(best)
        
        return outputs


# attention module
class Attention3D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(Attention3D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.comp_visual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.comp_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, visual_inputs, states):
        # visual_inputs = (batch_size, visual_flat, visual_channels)
        feature = visual_inputs.permute(0, 2, 1).contiguous()
        # get the hidden state
        hidden = states[0]
        # in = (batch_size, visual_flat, visual_channels)
        # out = (batch_size, visual_flat, hidden_size)
        V = self.comp_visual(feature)
        # in = (batch_size, hidden_size)
        # out = (batch_size, 1, hidden_size)
        H = self.comp_hidden(hidden).unsqueeze(1)
        # print("V", V.view(-1).min(0)[0].item(), V.view(-1).max(0)[0].item())
        # print("H", H.view(-1).min(0)[0].item(), H.view(-1).max(0)[0].item())
        # combine
        outputs = F.tanh(V + H)
        # outputs = (batch_size, visual_flat)
        outputs = self.output_layer(outputs).squeeze(2)
        outputs = F.softmax(outputs, dim=1)

        return outputs


# decoder with attention
class AttentionDecoder3D(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, visual_channels, visual_size, cuda_flag=True):
        super(AttentionDecoder3D, self).__init__()
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
        self.cuda_flag = cuda_flag
        # layer settings
        # initialize hidden states
        self.init_h = nn.Sequential(
            nn.Linear(self.visual_channels, hidden_size),
        )
        self.init_c = nn.Sequential(
            nn.Linear(self.visual_channels, hidden_size),
        )
        # embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        # attention layer
        self.attention = Attention3D(self.visual_channels, self.hidden_size, self.visual_flat)

        # LSTM
        self.lstm_layer_1 = nn.LSTMCell(2 * self.hidden_size, self.hidden_size)

        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size, self.input_size),
            # nn.Dropout(p=0.2)
        )


    def init_hidden(self, visual_inputs):
        visual_flat = visual_inputs.view(visual_inputs.size(0), visual_inputs.size(1), visual_inputs.size(2) * visual_inputs.size(2) * visual_inputs.size(2))
        visual_flat = visual_flat.mean(2)
        states = (
            self.init_h(visual_flat),
            self.init_c(visual_flat)
        )

        return states

    def forward(self, features, caption_inputs, states):
        _, global_features, area_features = features
        # feed
        seq_length = caption_inputs.size(1)
        decoder_outputs = []
        for step in range(seq_length):
            embedded = self.embedding(caption_inputs[:, step])
            lstm_input = torch.cat((embedded, global_features), dim=1)
            states = self.lstm_layer_1(lstm_input, states)
            lstm_outputs = states[0]
            attention_weights = self.attention(area_features, states)
            attended = torch.sum(area_features * attention_weights.unsqueeze(1), 2)
            outputs = torch.cat((attended, lstm_outputs), dim=1)
            outputs = self.output_layer(outputs).unsqueeze(1)
            decoder_outputs.append(outputs)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs 

    def sample(self, features, caption_inputs, states):
        _, global_features, area_features = features
        embedded = self.embedding(caption_inputs)
        lstm_input = torch.cat((embedded, global_features), dim=1)
        new_states = self.lstm_layer_1(lstm_input, states)
        lstm_outputs = new_states[0]
        attention_weights = self.attention(area_features, states)
        attended = torch.sum(area_features * attention_weights.unsqueeze(1), 2)
        outputs = torch.cat((attended, lstm_outputs), dim=1)
        outputs = self.output_layer(outputs).unsqueeze(1)

        return outputs, new_states, attention_weights

    def beam_search(self, features, caption_inputs, beam_size, max_length):
        batch_size = features[0].size(0)
        outputs = []
        for feat_id in range(batch_size):
            feats = (
                features[0][feat_id].unsqueeze(0),
                features[1][feat_id].unsqueeze(0),
                features[2][feat_id].unsqueeze(0),
            )
            states = self.init_hidden(feats[0])
            start, states, _ = self.sample(feats, caption_inputs[feat_id, 0].view(1), states)
            start = F.log_softmax(start, dim=2)
            start_scores, start_words = start.topk(beam_size, dim=2)[0].squeeze(), start.topk(beam_size, dim=2)[1].squeeze()
            # a queue containing all searched words and their log_prob
            searched = deque([([start_words[i].view(1)], start_scores[i].view(1), states) for i in range(beam_size)])
            done = []
            while True:
                candidate = searched.popleft()
                prev_word, prev_prob, prev_states = candidate
                if len(prev_word) <= max_length and int(prev_word[-1].item()) != 3:
                    preds, new_states, _ = self.sample(feats, prev_word[-1], prev_states)
                    preds = F.log_softmax(preds, dim=2)
                    top_scores, top_words = preds.topk(beam_size, dim=2)[0].squeeze(), preds.topk(beam_size, dim=2)[1].squeeze()
                    for i in range(beam_size):
                        next_word, next_prob = copy.deepcopy(prev_word), prev_prob.clone()
                        next_word.append(top_words[i].view(1))
                        next_prob += top_scores[i].view(1)
                        searched.append((next_word, next_prob, new_states))
                    searched = deque(sorted(searched, reverse=True, key=lambda s: s[1])[:beam_size])
                else:
                    done.append((prev_word, prev_prob))
                if not searched:
                    break           
            
            done = sorted(done, reverse=True, key=lambda s: s[1])
            best = [word[0].item() for word in done[0][0]]
            outputs.append(best)
        
        return outputs
