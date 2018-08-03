import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import lib.configs as configs
import nrrd
from itertools import combinations

class Shapenet():
    def __init__(self, shapenet_split, size_split, batch_size, is_training):
        '''
        param: shapenet_split: [shapenet_split_train, shapenet_split_val, shapenet_split_test]
        '''
        self.shapenet_split_train, self.shapenet_split_val, self.shapenet_split_test = shapenet_split
        self.train_size, self.val_size, self.test_size = size_split
        self.batch_size = batch_size

        # select sets
        if self.train_size != -1:
            self.shapenet_split_train = self.shapenet_split_train[:self.train_size]
        if self.val_size != -1:
            self.shapenet_split_val = self.shapenet_split_val[:self.val_size]
        if self.test_size != -1:
            self.shapenet_split_test = self.shapenet_split_test[:self.test_size]

        self.dict_idx2word, self.dict_word2idx = {}, {}
        self.train_data, self.val_data, self.test_data = [], [], []
        self._build_mapping()
        self._build_dict()
        self._transform()
        if is_training:
            self._aggregate()

    def _build_mapping(self):
        '''
        create mapping between model_ids and labels
        '''
        setattr(self, "cat2label", {'table': -1, 'chair': 1})
        for phase in ["train", "val", "test"]:
            idx2label = {}
            for label, item in enumerate(getattr(self, "shapenet_split_{}".format(phase))):
                model_id = item[0]
                if model_id not in idx2label.keys():
                    idx2label[model_id] = str(label)
            
            label2idx = {label: idx for idx, label in idx2label.items()}
            setattr(self, "{}_idx2label".format(phase), idx2label)
            setattr(self, "{}_label2idx".format(phase), label2idx)

    def _build_dict(self):
        '''
        create dictionaries
        '''
        split_data = self.shapenet_split_train
        word_count = {}
        for item in split_data:
            for word in item[2]:
                if word in word_count.keys():
                    word_count[word] += 1
                else:
                    word_count[word] = 1
        word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        # indexing starts at 4
        self.dict_word2idx = {word_count[i][0]: str(i + 4) for i in range(len(word_count))}
        self.dict_idx2word = {str(i + 4): word_count[i][0] for i in range(len(word_count))}
        # add special tokens
        self.dict_word2idx["<PAD>"] = str(0)
        self.dict_idx2word[str(0)] = "<PAD>"
        self.dict_word2idx["<UNK>"] = str(1)
        self.dict_idx2word[str(1)] = "<UNK>"
        self.dict_word2idx["<START>"] = str(2)
        self.dict_idx2word[str(2)] = "<START>"
        self.dict_word2idx["<END>"] = str(3)
        self.dict_idx2word[str(3)] = "<END>"
    
    def _transform(self):
        '''
        tokenize captions
        '''
        for phase in ["train", "val", "test"]:
            split_data = getattr(self, "shapenet_split_{}".format(phase))
            transformed = []
            for item in split_data:
                # get model_id
                model_id = item[0]
                # get label
                label = self.cat2label[item[1]]
                # truncate long captions
                words = item[2]
                if len(words) > configs.MAX_LENGTH:
                    words = words[:configs.MAX_LENGTH]
                indices = []
                # encode
                for word in words:
                    if word in self.dict_word2idx.keys():
                        indices.append(int(self.dict_word2idx[word]))
                    else:
                        indices.append(int(self.dict_word2idx["<UNK>"]))
                indices = [int(self.dict_word2idx["<START>"])] + indices + [int(self.dict_word2idx["<END>"])]
                # load into result
                transformed.append((model_id, label, indices))
            setattr(self, "{}_data".format(phase), transformed)
            setattr(self, "{}_size".format(phase), len(transformed))
    
    def _aggregate(self):
        '''
        aggregate data pairs such that:
        1. they are not the same caption.
        2. they correspond to the same model.
        3. there are no other captions in the batch that corresopnd to the same model.

        this method is only performed in training/validation step
        '''
        for phase in ["train", "val", "test"]:
            split_data = getattr(self, "{}_data".format(phase))
            
            # aggregate by model_id
            data_agg = {}
            for item in split_data:
                if item[0] in data_agg.keys():
                    data_agg[item[0]].append(item)
                else:
                    data_agg[item[0]] = [item]

            # get all combinations
            data_comb = []
            for key in data_agg.keys():
                data_comb.extend(list(combinations(data_agg[key], configs.N_CAPTION_PER_MODEL)))

            # aggregate batch
            data = []
            idx2label = {i: data_comb[i][0][0] for i in range(len(data_comb))}
            chosen_idx = []
            while len(data) < configs.N_CAPTION_PER_MODEL * len(data_comb):
                if len(chosen_idx) == self.batch_size:
                    chosen_idx = []
                idx = np.random.randint(len(data_comb))
                if idx2label[idx] in chosen_idx:
                    continue
                else:
                    data.extend([data_comb[idx][i] for i in range(configs.N_CAPTION_PER_MODEL)])
                    chosen_idx.append(idx)
            
            setattr(self, "{}_data".format(phase), data)



class ShapenetDataset(Dataset):
    def __init__(self, shapenet_data, idx2label, resolution):
        '''
        param: shapenet_data: instance property of Shapenet class, e.g. shapenet.train_data
        '''
        self.shapenet_data = shapenet_data
        self.idx2label = idx2label
        self.resolution = resolution

    def __len__(self):
        return len(self.shapenet_data)

    def __getitem__(self, idx):
        model_id = self.shapenet_data[idx][0]
        model_path = os.path.join(configs.SHAPE_ROOT.format(self.resolution), configs.SHAPENET_NRRD.format(model_id, model_id))
        voxel = torch.FloatTensor(nrrd.read(model_path)[0])
        voxel /= 255.
        caption = self.shapenet_data[idx][2]
        length = len(caption)
        label = int(self.idx2label[self.shapenet_data[idx][0]])

        return model_id, voxel, caption, length, label

def collate_shapenet(data):
    '''
    for DataLoader: collate_fn=collate_shapenet

    return: 
        model_ids
        5D tensor of voxels
        2D tensor of transformed captions
        lengths of transformed captions
        labels, table is -1 and chair is 1
    '''
    # # Sort a data list by caption length (descending order)
    # data.sort(key=lambda x: x[3], reverse=True)

    # Merge voxels (from tuple of 4D tensor to 5D tensor)
    model_ids, voxels, captions, lengths, labels = zip(*data)
    voxels = torch.stack(voxels, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    merge_caps = torch.zeros(len(captions), configs.MAX_LENGTH + 2).long()
    for i, cap in enumerate(captions):
        end = int(lengths[i])
        merge_caps[i, :end] = torch.LongTensor(cap[:end])
    
    return model_ids, voxels, merge_caps, torch.Tensor(list(lengths)), torch.Tensor(list(labels))