import torch
import pandas
import os
import re
import warnings
import operator
import copy
import nrrd
import math
import h5py
import pickle
import random
import string
import json
import numpy as np
from PIL import Image
from lib.configs import CONF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# for EmbeddingCaptionDataset
def collate_ec(data):
    # Sort a data list by caption length (descending order)
    data.sort(key=lambda x: x[4], reverse=True)

    # Merge embeddings (from tuple of 1D tensor to 2D tensor)
    model_id, model_cap, model_emb, model_interm_feat, lengths = zip(*data)
    model_emb = torch.stack(model_emb, 0)
    model_interm_feat = torch.stack(model_interm_feat, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    merge_cap = torch.zeros(len(model_cap), max(lengths)).long()
    for i, cap in enumerate(model_cap):
        end = int(lengths[i])
        merge_cap[i, :end] = torch.LongTensor(cap[:end])

    return model_id, merge_cap, model_emb, model_interm_feat, torch.Tensor(list(lengths))

class PretrainedEmbeddings():
    def __init__(self, pretrained_embeddings):
        '''
        params:
            pretrained_embeddings: [pretrained_train_pickle, pretrained_val_pickle, pretrained_test_pickle]
            size: [train_size, val_size, test_size]
            dict_idx2word: original vocabulary
            max_length: max length for truncation
        '''
        self.pretrained_embeddings = pretrained_embeddings
        # objectives
        self.train_text, self.val_text, self.test_text = [], [], []
        self.train_ref, self.val_ref, self.test_ref = {}, {}, {}
        self.train_shape, self.val_shape, self.test_shape = {}, {}, {}
        self.train_size, self.val_size, self.test_size = 0, 0, 0
        self.visual_channel, self.visual_size = 0, 0
        # decode tokenized captions
        self._decode()
        # build dict
        self.dict_idx2word, self.dict_word2idx  = self._build_dict()
        # transform
        self._transform()
        # get visual size
        self._get_visual_info()

    def _get_visual_info(self):
        data = getattr(self, "train_shape")
        example_id = list(data.keys())[0]
        setattr(self, "visual_channel", data[example_id][1].shape[0])
        setattr(self, "visual_size", data[example_id][1].shape[1])

    def _decode(self):
        # decode
        for phase in ['train', 'val', 'test']:
            pretrained_embeddings = self.pretrained_embeddings[phase]
            model_ids = list(pretrained_embeddings.keys())
            decoded = []
            feat = {}
            ref = {}
            for model_id in model_ids:
                feat[model_id] = (
                    pretrained_embeddings[model_id]['shape_embedding'][0],
                    pretrained_embeddings[model_id]['shape_embedding'][1]
                )
                for emb_id in range(len(pretrained_embeddings[model_id]['text_embedding'])):
                    cap = pretrained_embeddings[model_id]['text_embedding'][emb_id][0]
                    if len(cap) > CONF.CAP.MAX_LENGTH:
                        cap = cap[:CONF.CAP.MAX_LENGTH]
                    cap = '<START> ' + cap + ' <END>'
                    decoded.append(
                        [
                            model_id,  
                            cap
                        ]
                    )
                    if model_id in ref.keys():
                        ref[model_id].append(cap)
                    else:
                        ref[model_id] = [cap]
            
            setattr(self, "{}_text".format(phase), decoded)
            setattr(self, "{}_ref".format(phase), ref)
            setattr(self, "{}_shape".format(phase), feat)
            setattr(self, "{}_size".format(phase), len(decoded))

    def _build_dict(self):
        word_list = {}
        for item in getattr(self, "train_text"):
            for word in item[1].split(" "):
                if word in word_list.keys() and word != '<START>' and word != '<END>':
                    word_list[word] += 1
                else:
                    word_list[word] = 1
        
        # sort word in descending order
        word_list = sorted(word_list.items(), key=operator.itemgetter(1), reverse=True)
        word_list = [item[0] for item in word_list if item[0]]
        # build dict
        dict_word2idx = {word_list[i]: str(i + 4) for i in range(len(word_list))}
        dict_idx2word = {str(i + 4): word_list[i] for i in range(len(word_list))}
        # add special tokens
        dict_word2idx["<PAD>"] = str(0)
        dict_idx2word[str(0)] = "<PAD>"
        dict_word2idx["<UNK>"] = str(1)
        dict_idx2word[str(1)] = "<UNK>"
        dict_word2idx["<START>"] = str(2)
        dict_idx2word[str(2)] = "<START>"
        dict_word2idx["<END>"] = str(3)
        dict_idx2word[str(3)] = "<END>"
        
        return dict_idx2word, dict_word2idx

    def _transform(self):
        # transform
        for phase in ['train', 'val', 'test']:
            data = getattr(self, "{}_text".format(phase))
            for i in range(len(data)):
                cap_trans = []
                for word in data[i][1].split(' '):
                    if word in self.dict_word2idx.keys():
                        cap_trans.append(int(self.dict_word2idx[word]))
                    else:
                        cap_trans.append(int(self.dict_word2idx["<UNK>"]))
                data[i][1] = cap_trans
            
            # dump
            setattr(self, "{}_text".format(phase), data)
    

class CaptionDataset(Dataset):
    def __init__(self, text_set, shape_set):
        super(CaptionDataset, self).__init__()
        self.text_set = text_set
        self.shape_set = shape_set

    def __len__(self):
        return len(self.text_set)
    
    def __getitem__(self, idx):
        model_id = self.text_set[idx][0]
        model_cap = torch.LongTensor(self.text_set[idx][1])
        model_emb = torch.FloatTensor(self.shape_set[model_id][0])
        model_interm_feat = torch.FloatTensor(self.shape_set[model_id][1])
        length = len(model_cap)
        
        return model_id, model_cap, model_emb, model_interm_feat, length