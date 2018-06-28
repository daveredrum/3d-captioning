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
                        

