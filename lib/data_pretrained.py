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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# for EmbeddingCaptionDataset
def collate_ec(data):
    # Sort a data list by caption length (descending order)
    data.sort(key=lambda x: x[5], reverse=True)

    # Merge embeddings (from tuple of 1D tensor to 2D tensor)
    model_id, model_cat, model_cap, model_emb, model_interm, lengths = zip(*data)
    model_emb = torch.stack(model_emb, 0)
    model_interm = torch.stack(model_interm, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    merge_cap = torch.zeros(len(model_cap), max(lengths)).long()
    for i, cap in enumerate(model_cap):
        end = int(lengths[i])
        merge_cap[i, :end] = torch.LongTensor(cap[:end])

    return model_id, model_cat, merge_cap, model_emb, model_interm, torch.Tensor(list(lengths))

class PretrainedEmbeddings():
    def __init__(self, pretrained_embeddings, size, max_length):
        '''
        params:
            pretrained_embeddings: [pretrained_train_pickle, pretrained_val_pickle, pretrained_test_pickle]
            size: [train_size, val_size, test_size]
            dict_idx2word: original vocabulary
            max_length: max length for truncation
        '''
        self.pretrained_embeddings = pretrained_embeddings
        self.train_size, self.val_size, self.test_size = size
        self.max_length = max_length
        self.train_embeddings, self.val_embeddings, self.test_embeddings = None, None, None
        self.train_shape_embeddings, self.val_shape_embeddings, self.test_shape_embeddings = None, None, None
        self.train_ref, self.val_ref, self.test_ref = {}, {}, {}
        # decode tokenized captions
        self._decode()
        # build dict
        self.dict_idx2word, self.dict_word2idx  = self._build_dict()
        # transform
        self._transform()

    def _decode(self):
        # slice
        if self.train_size != -1:
            pretrained_train = self.pretrained_embeddings[0][:self.train_size]
        else:
            pretrained_train = self.pretrained_embeddings[0]
        if self.val_size != -1:
            pretrained_val = self.pretrained_embeddings[1][:self.val_size]
        else:
            pretrained_val = self.pretrained_embeddings[1]
        if self.test_size != -1:
            pretrained_test = self.pretrained_embeddings[2][:self.test_size]
        else:
            pretrained_test = self.pretrained_embeddings[2]
        
        # objectives
        self.train_embeddings, self.val_embeddings, self.test_embeddings = [], [], []
        self.train_shape_embeddings, self.val_shape_embeddings, self.test_shape_embeddings = {}, {}, {}
        
        # decode train
        for i in range(len(pretrained_train)):
            temp = pretrained_train[i][2]
            if len(temp) > self.max_length:
                temp = temp[:self.max_length]
            temp = ' '.join(temp)
            temp = '<START> ' + temp
            temp += ' <END>'
            self.train_embeddings.append([pretrained_train[i][0], pretrained_train[i][1], temp, pretrained_train[i][3], pretrained_train[i][4]])
            if pretrained_train[i][0] in self.train_ref.keys():
                self.train_ref[pretrained_train[i][0]].append(temp)
            else:
                self.train_ref[pretrained_train[i][0]] = [temp]
        self.train_shape_embeddings = {item[0]: (item[1], item[2], item[3], item[4]) for item in self.train_embeddings if item[0] not in self.train_shape_embeddings.keys()}
        self.train_shape_embeddings = [[item[0], item[1][0], item[1][1], item[1][2], item[1][3]] for item in self.train_shape_embeddings.items()]
        
        # decode val
        for i in range(len(pretrained_val)):
            temp = pretrained_val[i][2]
            if len(temp) > self.max_length:
                temp = temp[:self.max_length]
            temp = ' '.join(temp)
            temp = '<START> ' + temp
            temp += ' <END>'
            self.val_embeddings.append([pretrained_val[i][0], pretrained_val[i][1], temp, pretrained_val[i][3], pretrained_val[i][4]])
            if pretrained_val[i][0] in self.val_ref.keys():
                self.val_ref[pretrained_val[i][0]].append(temp)
            else:
                self.val_ref[pretrained_val[i][0]] = [temp]
        self.val_shape_embeddings = {item[0]: (item[1], item[2], item[3], item[4]) for item in self.val_embeddings if item[0] not in self.val_shape_embeddings.keys()}
        self.val_shape_embeddings = [[item[0], item[1][0], item[1][1], item[1][2], item[1][3]] for item in self.val_shape_embeddings.items()]
       
        # decode test
        for i in range(len(pretrained_test)):
            temp = pretrained_test[i][2]
            if len(temp) > self.max_length:
                temp = temp[:self.max_length]
            temp = ' '.join(temp)
            temp = '<START> ' + temp
            temp += ' <END>'
            self.test_embeddings.append([pretrained_test[i][0], pretrained_test[i][1], temp, pretrained_test[i][3], pretrained_test[i][4]])
            if pretrained_test[i][0] in self.test_ref.keys():
                self.test_ref[pretrained_test[i][0]].append(temp)
            else:
                self.test_ref[pretrained_test[i][0]] = [temp]
        self.test_shape_embeddings = {item[0]: (item[1], item[2], item[3], item[4]) for item in self.test_embeddings if item[0] not in self.test_shape_embeddings.keys()}
        self.test_shape_embeddings = [[item[0], item[1][0], item[1][1], item[1][2], item[1][3]] for item in self.test_shape_embeddings.items()]


    def _build_dict(self):
        word_list = {}
        for item in self.train_embeddings:
            for word in item[2].split(" "):
                if word in word_list.keys() and word != '<START>' and word != '<END>':
                    word_list[word] += 1
                else:
                    word_list[word] = 1
        word_list = sorted(word_list.items(), key=operator.itemgetter(1), reverse=True)
        word_list = [item[0] for item in word_list]
        # indexing starts at 4
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
        # transform train
        for i in range(len(self.train_embeddings)):
            temp = []
            for word in self.train_embeddings[i][2].split(" "):
                if word in self.dict_word2idx.keys():
                    temp.append(int(self.dict_word2idx[word]))
                else:
                    temp.append(int(self.dict_word2idx['<UNK>']))
            self.train_embeddings[i][2] = temp
        for i in range(len(self.train_shape_embeddings)):
            temp = []
            for word in self.train_shape_embeddings[i][2].split(" "):
                if word in self.dict_word2idx.keys():
                    temp.append(int(self.dict_word2idx[word]))
                else:
                    temp.append(int(self.dict_word2idx['<UNK>']))
            self.train_shape_embeddings[i][2] = temp
        
        # transform val
        for i in range(len(self.val_embeddings)):
            temp = []
            for word in self.val_embeddings[i][2].split(" "):
                if word in self.dict_word2idx.keys():
                    temp.append(int(self.dict_word2idx[word]))
                else:
                    temp.append(int(self.dict_word2idx['<UNK>']))
            self.val_embeddings[i][2] = temp
        for i in range(len(self.val_shape_embeddings)):
            temp = []
            for word in self.val_shape_embeddings[i][2].split(" "):
                if word in self.dict_word2idx.keys():
                    temp.append(int(self.dict_word2idx[word]))
                else:
                    temp.append(int(self.dict_word2idx['<UNK>']))
            self.val_shape_embeddings[i][2] = temp
       
        # transform test
        for i in range(len(self.test_embeddings)):
            temp = []
            for word in self.test_embeddings[i][2].split(" "):
                if word in self.dict_word2idx.keys():
                    temp.append(int(self.dict_word2idx[word]))
                else:
                    temp.append(int(self.dict_word2idx['<UNK>']))
            self.test_embeddings[i][2] = temp
        for i in range(len(self.test_shape_embeddings)):
            temp = []
            for word in self.test_shape_embeddings[i][2].split(" "):
                if word in self.dict_word2idx.keys():
                    temp.append(int(self.dict_word2idx[word]))
                else:
                    temp.append(int(self.dict_word2idx['<UNK>']))
            self.test_shape_embeddings[i][2] = temp
    

class EmbeddingCaptionDataset(Dataset):
    def __init__(self, pretrained_embeddings):
        super(EmbeddingCaptionDataset, self).__init__()
        self.pretrained_embeddings = pretrained_embeddings

    def __len__(self):
        return len(self.pretrained_embeddings)
    
    def __getitem__(self, idx):
        model_id = self.pretrained_embeddings[idx][0]
        model_cat = self.pretrained_embeddings[idx][1]
        model_cap = torch.LongTensor(self.pretrained_embeddings[idx][2])
        model_emb = torch.Tensor(self.pretrained_embeddings[idx][3])
        model_interm = torch.Tensor(self.pretrained_embeddings[idx][4])
        length = len([item for item in model_cap if item != 0])
        
        return model_id, model_cat, model_cap, model_emb, model_interm, length