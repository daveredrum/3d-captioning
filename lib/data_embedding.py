import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import nrrd
from itertools import combinations
import random
import h5py
import time
import copy
import random
from collections import deque

# HACK
import sys
sys.path.append(".")
from lib.configs import CONF


class Embedding():
    def __init__(self, data_split, size_split, batch_size, is_training):
        '''
        param: 
            data_split: [data_split_train, data_split_val, data_split_test]
            size_split: [size_split_train, size_split_val, size_split_test]
            batch_size: unique_batch_size
            is_training: boolean, distinguishing between train/val and eval
        '''
        
        # general settings
        self.data_split_train, self.train_modelid2idx = data_split[0]['data'], data_split[0]['modelid2idx']
        self.data_split_val, self.val_modelid2idx = data_split[1]['data'], data_split[1]['modelid2idx']
        self.data_split_test, self.test_modelid2idx = data_split[2]['data'], data_split[2]['modelid2idx']
        self.train_size, self.val_size, self.test_size = size_split
        self.batch_size = batch_size
        self.bad_ids = pickle.load(open(CONF.PATH.SHAPENET_PROBLEMATIC, 'rb'))

        # select sets
        if self.train_size != -1:
            self.data_split_train = self.data_split_train[:self.train_size]
        if self.val_size != -1:
            self.data_split_val = self.data_split_val[:self.val_size]
        if self.test_size != -1:
            self.data_split_test = self.data_split_test[:self.test_size]

        # objectives
        self.dict_idx2word, self.dict_word2idx = {}, {}
        self.train_data, self.val_data, self.test_data = [], [], []
        self.train_data_group, self.val_data_group, self.test_data_group = {}, {}, {}
        self.train_data_agg, self.val_data_agg, self.test_data_agg = {}, {}, {}
        self.train_idx2label, self.val_idx2label, self.test_idx2label = {}, {}, {}
        self.train_label2idx, self.val_label2idx, self.test_label2idx = {}, {}, {}

        # execute
        self._build_mapping()
        self._build_dict()
        self._transform()
        if is_training:
            self._aggregate()

    def _build_mapping(self):
        '''
        create mapping between model_ids and labels
        '''
        for phase in ["train", "val", "test"]:
            idx2label = {idx: label for idx, label in getattr(self, "{}_modelid2idx".format(phase)).items()}
            label2idx = {label: idx for idx, label in getattr(self, "{}_modelid2idx".format(phase)).items()}
            setattr(self, "{}_idx2label".format(phase), idx2label)
            setattr(self, "{}_label2idx".format(phase), label2idx)

    def _build_dict(self):
        '''
        create dictionaries
        '''
        split_data = self.data_split_train
        word_count = {}
        for item in split_data:
            for word in item[2]:
                if word in word_count.keys():
                    word_count[word] += 1
                else:
                    word_count[word] = 1
        word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        # indexing starts at 2
        self.dict_word2idx = {word_count[i][0]: str(i + 2) for i in range(len(word_count))}
        self.dict_idx2word = {str(i + 2): word_count[i][0] for i in range(len(word_count))}
        # add special tokens
        self.dict_word2idx["<PAD>"] = str(0)
        self.dict_idx2word[str(0)] = "<PAD>"
        self.dict_word2idx["<UNK>"] = str(1)
        self.dict_idx2word[str(1)] = "<UNK>"
        # # indexing starts at 4
        # self.dict_word2idx = {word_count[i][0]: str(i + 4) for i in range(len(word_count))}
        # self.dict_idx2word = {str(i + 4): word_count[i][0] for i in range(len(word_count))}
        # # add special tokens
        # self.dict_word2idx["<PAD>"] = str(0)
        # self.dict_idx2word[str(0)] = "<PAD>"
        # self.dict_word2idx["<UNK>"] = str(1)
        # self.dict_idx2word[str(1)] = "<UNK>"
        # self.dict_word2idx["<START>"] = str(2)
        # self.dict_idx2word[str(2)] = "<START>"
        # self.dict_word2idx["<END>"] = str(3)
        # self.dict_idx2word[str(3)] = "<END>"
    
    def _transform(self):
        '''
        tokenize captions
        '''
        for phase in ["train", "val", "test"]:
            split_data = getattr(self, "data_split_{}".format(phase))
            transformed = []
            data_group = {}
            for item in split_data:
                # get model_id
                model_id = item[0]
                if model_id in self.bad_ids:
                    continue
                # get label
                label = item[1]
                # truncate long captions
                words = item[2]
                if len(words) > CONF.TRAIN.MAX_LENGTH:
                    words = words[:CONF.TRAIN.MAX_LENGTH]
                indices = []
                # encode
                for word in words:
                    if word in self.dict_word2idx.keys():
                        indices.append(int(self.dict_word2idx[word]))
                    else:
                        indices.append(int(self.dict_word2idx["<UNK>"]))

                # indices = [int(self.dict_word2idx["<START>"])] + indices + [int(self.dict_word2idx["<END>"])]
                # load into result
                transformed.append((model_id, label, indices))

                # group by key
                if CONF.TRAIN.DATASET == 'shapenet':
                    if model_id in data_group.keys():
                        data_group[model_id].append((model_id, label, indices))
                    else:
                        data_group[model_id] = [(model_id, label, indices)]
                elif CONF.TRAIN.DATASET == 'primitives':
                    if label in data_group.keys():
                        data_group[label].append((model_id, label, indices))
                    else:
                        data_group[label] = [(model_id, label, indices)]
                else:
                    raise ValueError("invalid dataset, terminating...")

            setattr(self, "{}_data".format(phase), transformed)
            setattr(self, "{}_data_group".format(phase), data_group)
            setattr(self, "{}_size".format(phase), len(data_group.keys()))
    
    def _aggregate(self):
        '''
        aggregate data pairs such that:
        1. they are not the same caption.
        2. they correspond to the same model.
        3. there are no other captions in the batch that corresopnd to the same model.

        this method is only performed in training/validation step
        '''
        for phase in ["train", "val", "test"]:
            group_data = getattr(self, "{}_data_group".format(phase))
            if CONF.TRAIN.DATASET == 'shapenet':
                # get all combinations
                data_comb = []
                for key in group_data.keys():
                    if len(group_data[key]) >= 4:
                        comb = list(combinations(group_data[key], CONF.TRAIN.N_CAPTION_PER_MODEL))
                        random.shuffle(comb)
                        data_comb.extend(comb)

                # aggregate batch
                data = []
                idx2label = {i: data_comb[i][0][0] for i in range(len(data_comb))}
                chosen_label = []
                while len(data) < CONF.TRAIN.N_CAPTION_PER_MODEL * len(data_comb):
                    if len(chosen_label) == self.batch_size:
                        chosen_label = []
                    idx = np.random.randint(len(data_comb))
                    if idx2label[idx] in chosen_label:
                        continue
                    else:
                        data.extend([data_comb[idx][i] for i in range(CONF.TRAIN.N_CAPTION_PER_MODEL)])
                        chosen_label.append(idx2label[idx])
                
                setattr(self, "{}_data_agg".format(phase), data)
            elif CONF.TRAIN.DATASET == 'primitives':
                # group by label within modelid
                new_group_data = {}
                for model_id in group_data.keys():
                    grouped_label = {}
                    for item in group_data[model_id]:
                        if item[0] in grouped_label.keys():
                            grouped_label[item[0]].append(item)
                        else:
                            grouped_label[item[0]] = [item]
                    new_group_data[model_id] = grouped_label
                group_data = new_group_data
                # get all combinations
                for model_id in group_data.keys():
                    for label in group_data[model_id].keys():
                        comb = list(combinations(group_data[model_id][label], CONF.TRAIN.N_CAPTION_PER_MODEL))
                        random.shuffle(comb)
                        group_data[model_id][label] = comb

                all_model_ids = list(group_data.keys())
                # aggregate batch
                data = []
                for label_id in range(CONF.TRAIN.PRIMITIVES_NUM_PER_MODEL):
                    num_pairs = int((CONF.TRAIN.PRIMITIVES_NUM_PER_MODEL // 2) * (CONF.TRAIN.PRIMITIVES_NUM_PER_MODEL - 1))
                    for pair_id in range(num_pairs):
                        num_rand = len(all_model_ids) // self.batch_size
                        if num_rand < 1:
                            num_rand = 1
                        for _ in range(num_rand):
                            chosen_model_ids = random.sample(all_model_ids, k=self.batch_size)
                            for model_id in chosen_model_ids:
                                data.extend([group_data[model_id]["{}_{}".format(model_id, label_id)][pair_id][i] for i in range(CONF.TRAIN.N_CAPTION_PER_MODEL)])
                
                setattr(self, "{}_data_agg".format(phase), data)



class EmbeddingDataset(Dataset):
    def __init__(self, embedding_data, idx2label, label2idx, resolution, database=None, aggr_shape=False):
        '''
        param: 
            embedding_data: instance property of Shapenet class, e.g. embedding.train_data
        '''
        self.embedding_data = copy.deepcopy(embedding_data)
        if aggr_shape:
            self.embedding_data = self._aggr_data()
        self.idx2label = idx2label
        self.label2idx = label2idx
        self.resolution = resolution
        self.database = database

    def _aggr_data(self):
        aggr_data = []
        reached_model = []
        for item in self.embedding_data:
            if item[0] not in reached_model:
                aggr_data.append(item)
                reached_model.append(item[0])
        
        return aggr_data

    def __len__(self):
        return len(self.embedding_data)

    def __getitem__(self, idx):
        start = time.time()
        model_id = self.embedding_data[idx][0]
        label = self.embedding_data[idx][1]
        caption = self.embedding_data[idx][2]
        length = len(caption)
        if CONF.TRAIN.DATASET == 'shapenet':
            if self.database:
                db_idx = self.idx2label[model_id]
                voxel = self.database['volume'][db_idx].reshape((4, self.resolution, self.resolution, self.resolution))
                voxel = torch.FloatTensor(voxel)
            else:
                model_path = os.path.join(CONF.PATH.SHAPENET_ROOT.format(self.resolution), CONF.PATH.SHAPENET_NRRD.format(model_id, model_id))
                voxel = torch.FloatTensor(nrrd.read(model_path)[0])
                voxel /= 255.

            label = int(self.idx2label[model_id])
        elif CONF.TRAIN.DATASET == 'primitives':
            model_path = os.path.join(CONF.PATH.PRIMITIVES_ROOT.format(self.resolution), CONF.PATH.PRIMITIVES_NRRD.format(label, model_id))
            voxel = torch.FloatTensor(nrrd.read(model_path)[0])
            voxel /= 255.

            label = int(self.idx2label[label])
       
        exe_time = time.time() - start

        return model_id, voxel, caption, length, label, exe_time

def collate_embedding(data):
    '''
    for DataLoader: collate_fn=collate_embedding

    return: 
        model_ids
        5D tensor of voxels
        2D tensor of transformed captions
        lengths of transformed captions
        labels
        data fetch time
    '''
    # # Sort a data list by caption length (descending order)
    # data.sort(key=lambda x: x[3], reverse=True)

    # Merge voxels (from tuple of 4D tensor to 5D tensor)
    model_ids, voxels, captions, lengths, labels, exe_time = zip(*data)
    voxels = torch.stack(voxels, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    merge_caps = torch.zeros(len(captions), CONF.TRAIN.MAX_LENGTH).long()
    for i, cap in enumerate(captions):
        end = int(lengths[i])
        merge_caps[i, :end] = torch.LongTensor(cap[:end])
    
    return model_ids, voxels, merge_caps, torch.Tensor(list(lengths)), torch.Tensor(list(labels)), np.sum(exe_time)
