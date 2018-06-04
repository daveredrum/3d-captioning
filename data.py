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

# image dataset for encoder
class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform

    def __len__(self):
        return self.csv_file.id.count()

    def __getitem__(self, index):
        # model
        model_name = self.csv_file.modelId[index]
        model_label = self.csv_file.category[index]
        model_path = os.path.join(self.root_dir, model_name, model_name + '.png')
        # label mapping
        label_map = {'Table': 0, 'Chair': 1}
        # load data
        image = Image.open(model_path)
        image = np.array(image)[:, :, :3]
        image = Image.fromarray(image)
        label = label_map[model_label]
        if self.transform:
            image = self.transform(image)
        
        return image, label

# shape dataset for 3d encoder
class ShapeDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.root_dir = root_dir
        self.csv_file = csv_file
    
    def __len__(self):
        return self.csv_file.id.count()

    def __getitem__(self, index):
        # model
        model_name = self.csv_file.modelId[index]
        model_label = self.csv_file.category[index]
        model_path = os.path.join(self.root_dir, model_name, model_name + '.nrrd')
        # label mapping
        label_map = {'Table': 0, 'Chair': 1}
        # load and normalize data
        shape, _ = nrrd.read(model_path)
        shape = shape[:3, :, :, :]
        shape = (shape - shape.min()) / (shape.max() - shape.min())
        shape = torch.FloatTensor(shape)
        label = label_map[model_label]
        
        return shape, label
    

# caption dataset for decoder
class CaptionDataset(Dataset):
    def __init__(self, visual_array, caption_list):
        # visual inputs and caption inputs must have the same length
        assert visual_array.shape[0] == len(caption_list)
        self.visual_array = visual_array
        self.caption_list = copy.deepcopy(caption_list)
        self.data_pairs = self._build_data_pairs()

    def _build_data_pairs(self):
        # initialize data pairs: (visual, caption, cap_length)
        data_pairs = [(
            self.visual_array[i], 
            self.caption_list[i], 
            len(self.caption_list[i])
            ) for i in range(self.__len__())]
        # # # sort data pairs according to cap_length in descending order
        # data_pairs = sorted(data_pairs, key=lambda item: len(item[1]), reverse=True)
        # pad caption with 0 if it's length is not maximum
        for index in range(1, len(data_pairs)):
            for i in range(len(data_pairs[0][1]) - len(data_pairs[index][1])):
                data_pairs[index][1].append(0)

        return data_pairs

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, idx):
        # return (visual, caption_inputs, caption_targets, cap_length)
        return self.data_pairs[idx]

# dataset for coco
class COCOCaptionDataset(Dataset):
    def __init__(self, index_path, csv_file, database):
        self.index = json.load(open(index_path, "r"))
        self.model_ids = copy.deepcopy(csv_file.image_id.values.tolist())
        self.caption_lists = copy.deepcopy(csv_file.caption.values.tolist())
        self.csv_file = copy.deepcopy(csv_file)
        self.data_pairs = self._build_data_pairs()
        self.database = h5py.File(database, "r")

    def _build_data_pairs(self):
        # initialize data pairs: (model_id, image_path, caption, cap_length)
        data_pairs = [(
            str(self.model_ids[i]),
            self.index[str(i)],
            self.caption_lists[i],
            len(self.caption_lists[i])
        ) for i in range(self.__len__())]
        # sort data pairs according to cap_length in descending order
        data_pairs = sorted(data_pairs, key=lambda item: item[3], reverse=True)
        # pad caption with 0 if it's length is not maximum
        for index in range(1, len(data_pairs)):
            for i in range(len(data_pairs[0][2]) - len(data_pairs[index][2])):
                data_pairs[index][2].append(0)
        
        return data_pairs

    def __len__(self):
        return self.csv_file.image_id.count()

    def __getitem__(self, idx):
        # return (model_id, image_inputs, padded_caption, cap_length)
        image = self.database["features"][self.data_pairs[idx][1]]
        image = np.reshape(image, (512, 14, 14))
        image = torch.FloatTensor(image)

        return self.data_pairs[idx][0], image, self.data_pairs[idx][2], self.data_pairs[idx][3]

class FeatureDataset(Dataset):
    def __init__(self, database):
        self.database = h5py.File(database, "r")

    def __len__(self):
        return self.database["images"].shape[0]

    def __getitem__(self, idx):
        image = self.database["images"][idx]
        image = torch.FloatTensor(image).view(3, 224, 224)

        return image

# pipeline dataset for the encoder-decoder of image-caption
class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, csv_file, database):
        self.model_ids = copy.deepcopy(csv_file.modelId.values.tolist())
        self.caption_lists = copy.deepcopy(csv_file.description.values.tolist())
        self.csv_file = copy.deepcopy(csv_file)
        self.data_pairs = self._build_data_pairs()
        self.database = h5py.File(database, "r")

    def _build_data_pairs(self):
        # initialize data pairs: (model_id, image_path, caption, cap_length)
        data_pairs = [(
            self.model_ids[i],
            i,
            self.caption_lists[i],
            len(self.caption_lists[i])
        ) for i in range(self.__len__())]
        # sort data pairs according to cap_length in descending order
        data_pairs = sorted(data_pairs, key=lambda item: item[3], reverse=True)
        # pad caption with 0 if it's length is not maximum
        for index in range(1, len(data_pairs)):
            for i in range(len(data_pairs[0][2]) - len(data_pairs[index][2])):
                data_pairs[index][2].append(0)
        
        return data_pairs

    def __len__(self):
        return self.csv_file.id.count()

    def __getitem__(self, idx):
        # return (model_id, image_inputs, padded_caption, cap_length)
        image = self.database["images"][self.data_pairs[idx][1]]
        size = int(np.sqrt(image.shape[0] / 3))
        image = np.reshape(image, (3, size, size))
        image = torch.FloatTensor(image)
        # image = np.array(image)[:, :, :3]
        # image = Image.fromarray(image)
        # if self.transform:
        #     image = self.transform(image)

        return self.data_pairs[idx][0], image, self.data_pairs[idx][2], self.data_pairs[idx][3]

# pipeline dataset for the encoder-decoder of shape-caption
# only two modes are available
# default: load the shape data directly
# hdf5: load the preprocessed data in hdf5 file, path to the database is required in this mode
class ShapeCaptionDataset(Dataset):
    def __init__(self, root_dir, csv_file, database):
        self.model_ids = copy.deepcopy(csv_file.modelId.values.tolist())
        self.image_paths = [
            os.path.join(root_dir, model_name, model_name + '.png') 
            for model_name in self.model_ids
        ]
        self.shape_paths = [
            os.path.join(root_dir, model_name, model_name + '.nrrd') 
            for model_name in self.model_ids
        ]
        self.caption_lists = copy.deepcopy(csv_file.description.values.tolist())
        self.csv_file = copy.deepcopy(csv_file)
        self.data_pairs = self._build_data_pairs()
        self.database = h5py.File(database, "r")

     # initialize data pairs: (model_id, image_path, shape_path, caption, cap_length)
    def _build_data_pairs(self):
        data_pairs = [(
            self.model_ids[i],
            self.image_paths[i],
            i,
            self.caption_lists[i],
            len(self.caption_lists[i])
        ) for i in range(self.__len__())]
        # sort data pairs according to cap_length in descending order
        data_pairs = sorted(data_pairs, key=lambda item: item[4], reverse=True)
        # pad caption with 0 if it's length is not maximum
        for index in range(1, len(data_pairs)):
            for i in range(len(data_pairs[0][3]) - len(data_pairs[index][3])):
                data_pairs[index][3].append(0)
        
        return data_pairs

    def __len__(self):
        return self.csv_file.id.count()

    # 3d: (model_id, (image_path, shape_inputs), padded_caption, cap_length)
    def __getitem__(self, idx):
        # get images path
        image = self.data_pairs[idx][1]
        # get preprocessed shapes
        shape = np.array(self.database["shapes"][self.data_pairs[idx][2]])
        size = int(np.cbrt(shape.shape[0] / 3))
        shape = np.reshape(shape, (3, size, size, size))
        shape = torch.FloatTensor(shape)

        visual = (image, shape)

        return self.data_pairs[idx][0], visual, self.data_pairs[idx][3], self.data_pairs[idx][4]

# process shapenet csv file
class Caption(object):
    def __init__(self, csv_file, size_split):
        # size settings
        self.total_size = np.sum(size_split)
        self.train_size = size_split[0]
        self.val_size = size_split[1]
        self.test_size = size_split[2]
        # select data by the given total size
        self.original_csv = csv_file.iloc[:self.total_size]
        # preprocessed captions
        self.preprocessed_csv = None
        # captions translated to token indices
        self.transformed_csv = None
        # dictionaries
        self.dict_word2idx = None
        self.dict_idx2word = None
        self.dict_size = None
        # ground truth captions grouped by modelId
        # for calculating BLEU score
        # e.g. 'e702f89ce87a0b6579368d1198f406e7': [['<START> a gray coloured round four legged steel table <END>']]
        self.corpus = {
            'train': {},
            'val': {},
            'test': {}
        }
        # split the preprocessed data
        self.preprocessed_data = {
            'train': None,
            'val': None,
            'test': None
        }
        # split the transformed data
        self.transformed_data = {
            'train': None,
            'val': None,
            'test': None
        }
        
        # preprcess and transform
        self._preprocess()
        self._tranform()
        # build reference corpus
        self._make_corpus()

    # output the dictionary of captions
    # indices of words are the rank of frequencies
    def _make_dict(self):
        captions_list = self.preprocessed_csv.description.values.tolist()
        word_list = {}
        for text in captions_list:
            try:
                for word in re.split("[ ]", text):
                    if word and len(word_list.keys()) <= 5000:
                        if word in word_list.keys():
                            word_list[word] += 1
                        else:
                            word_list[word] = 1
            except Exception:
                pass
        # filter out all words that appear less than 5 times
        word_list = sorted(word_list.items(), key=operator.itemgetter(1), reverse=True)[:10000]
        # indexing starts at 1
        self.dict_word2idx = {word_list[i][0]: i+1 for i in range(len(word_list))}
        self.dict_idx2word = {i+1: word_list[i][0] for i in range(len(word_list))}
        # dictionary size
        assert self.dict_idx2word.__len__() == self.dict_word2idx.__len__()
        self.dict_size = self.dict_idx2word.__len__()
        
    # build the references for calculating BLEU score
    # return the dictionary of modelId and corresponding captions
    # input must be the preprocessed csv！
    def _make_corpus(self):
        for phase in ['train', 'val', 'test']:
            for _, item in self.preprocessed_data[phase].iterrows():
                if item.modelId in self.corpus[phase].keys():
                    self.corpus[phase][item.modelId].append(item.description)
                else:
                    self.corpus[phase][item.modelId] = [item.description]


    def _preprocess(self):
        # suppress all warnings
        warnings.simplefilter('ignore')
        # drop items without captions
        self.preprocessed_csv = copy.deepcopy(self.original_csv.loc[self.original_csv.description.notnull()].reset_index(drop=True))
        # convert to lowercase
        self.preprocessed_csv.description = self.preprocessed_csv.description.str.lower()
        # preprocess
        captions_list = self.preprocessed_csv.description.values.tolist()
        for i in range(len(captions_list)):
            # padding before all punctuations
            caption = captions_list[i]
            caption = re.sub(r'([.,!?()])', r' \1 ', caption)
            caption = re.sub(r'\s{2,}', ' ', caption)
            # add start symbol
            caption = '<START> ' + caption
            # add end symbol
            caption += '<END>'
            captions_list[i] = caption
            # filter out empty element
            caption = filter(None, caption)
        # replace with the new column
        new_captions = pandas.DataFrame({'description': captions_list})
        self.preprocessed_csv.description = new_captions.description
        # sort the csv file by the lengths of descriptions
        self.preprocessed_csv = self.preprocessed_csv.iloc[(-self.preprocessed_csv.description.str.len()).argsort()].reset_index(drop=True)

        # split the data
        self.preprocessed_data['train'] = self.preprocessed_csv.iloc[:self.train_size]
        self.preprocessed_data['val'] = self.preprocessed_csv.iloc[self.train_size:self.train_size + self.val_size]
        self.preprocessed_data['test'] = self.preprocessed_csv.iloc[self.train_size + self.val_size:]

        # build dict
        self._make_dict()

    # transform all words to their indices in the dictionary
    def _tranform(self):
        self.transformed_csv = copy.deepcopy(self.preprocessed_csv)
        captions_list = self.transformed_csv.description.values.tolist()
        for i in range(len(captions_list)):
            temp_list = []
            for text in captions_list[i].split(" "):
                # filter out empty element
                if text and text in self.dict_word2idx.keys():
                    temp_list.append(self.dict_word2idx[text])
                captions_list[i] = temp_list
        # replace with the new column
        transformed_captions = pandas.DataFrame({'description': captions_list})
        self.transformed_csv.description = transformed_captions.description
        # # sort the csv file by the lengths of descriptions
        # self.tranformed_csv = self.tranformed_csv.iloc[(-self.tranformed_csv.description.str.len()).argsort()].reset_index(drop=True)

        # split the data
        self.transformed_data['train'] = self.transformed_csv.iloc[:self.train_size]
        self.transformed_data['val'] = self.transformed_csv.iloc[self.train_size:self.train_size + self.val_size]
        self.transformed_data['test'] = self.transformed_csv.iloc[self.train_size + self.val_size:]

    # check if the transformation is reversable
    def sanity_check(self):
        captions_list = self.preprocessed_csv.description.values.tolist()
        reverse_list = self.transformed_csv.description.values.tolist()
        for i in range(len(reverse_list)):
            temp_string = ""
            for index in reverse_list[i]:
                temp_string += self.dict_idx2word[index]
                if index != reverse_list[i][-1]:
                    temp_string += " "
            reverse_list[i] = temp_string
        
        return reverse_list.sort() == captions_list.sort()

# process coco csv file
class COCO(object):
    def __init__(self, train_csv, val_csv, test_csv, size_split):
        # size settings
        self.total_size = np.sum(size_split)
        self.train_size, self.val_size, self.test_size = size_split
        # select data by the given total size
        self.original_csv = {
            'train': None,
            'val': None,
            'test': None
        }
        # use image_id to select data
        # training set
        if self.train_size == -1:
            self.original_csv['train'] = train_csv
        else:
            train_id = train_csv.image_id.drop_duplicates().values.tolist()[:self.train_size]
            self.original_csv['train'] = train_csv.loc[train_csv.image_id.isin(train_id)]
        # valation set
        if self.val_size == -1:
            self.original_csv['val'] = val_csv
        else:
            val_id = val_csv.image_id.drop_duplicates().values.tolist()[:self.val_size]
            self.original_csv['val'] = val_csv.loc[val_csv.image_id.isin(val_id)]
        # testing set
        if self.test_size == -1:
            self.original_csv['test'] = test_csv
        else:
            test_id = test_csv.image_id.drop_duplicates().values.tolist()[:self.test_size]
            self.original_csv['test'] = test_csv.loc[test_csv.image_id.isin(test_id)]
        # dictionaries
        self.dict_word2idx = None
        self.dict_idx2word = None
        self.dict_size = None
        # ground truth captions grouped by image_id
        # for calculating BLEU score
        self.corpus = {
            'train': {},
            'val': {},
            'test': {}
        }
        # split the preprocessed data
        self.preprocessed_data = {
            'train': None,
            'val': None,
            'test': None
        }
        # split the transformed data
        self.transformed_data = {
            'train': None,
            'val': None,
            'test': None
        }
        
        # preprcess and transform
        self._preprocess()
        self._tranform()
        # build reference corpus
        self._make_corpus()

    # output the dictionary of captions
    # indices of words are the rank of frequencies
    def _make_dict(self):
        captions_list = self.preprocessed_data["train"].caption.values.tolist()
        captions_list += self.preprocessed_data["val"].caption.values.tolist() 
        captions_list += self.preprocessed_data["test"].caption.values.tolist()
        word_list = {}
        for text in captions_list:
            try:
                for word in re.split("[ ]", text):
                    if word and word != "<START>" and word != "<END>":
                        # set the maximum size of vocabulary
                        if word in word_list.keys():
                            word_list[word] += 1
                        else:
                            word_list[word] = 1
            except Exception:
                pass
        # max dict_size = 10000
        word_list = sorted(word_list.items(), key=operator.itemgetter(1), reverse=True)[:10000]
        # indexing starts at 1
        self.dict_word2idx = {word_list[i][0]: i+3 for i in range(len(word_list))}
        self.dict_idx2word = {i+3: word_list[i][0] for i in range(len(word_list))}
        # add special tokens
        self.dict_word2idx["<UNK>"] = 0
        self.dict_idx2word[0] = "<UNK>"
        self.dict_word2idx["<START>"] = 1
        self.dict_idx2word[1] = "<START>"
        self.dict_word2idx["<END>"] = 2
        self.dict_idx2word[2] = "<END>"
        # dictionary size
        assert self.dict_idx2word.__len__() == self.dict_word2idx.__len__()
        self.dict_size = self.dict_idx2word.__len__()
        
    # build the references for calculating BLEU score
    # return the dictionary of image_id and corresponding captions
    # input must be the preprocessed csv！
    def _make_corpus(self):
        for phase in ["train", "val", "test"]:
            for _, item in self.preprocessed_data[phase].iterrows():
                if str(item.image_id) in self.corpus[phase].keys():
                    self.corpus[phase][str(item.image_id)].append(item.caption)
                else:
                    self.corpus[phase][str(item.image_id)] = [item.caption]


    def _preprocess(self):
        # suppress all warnings
        warnings.simplefilter('ignore')
        for phase in ["train", "val", "test"]:
            # drop items without captions
            self.preprocessed_data[phase] = copy.deepcopy(self.original_csv[phase].loc[self.original_csv[phase].caption.notnull()].reset_index(drop=True))
            # convert to lowercase
            self.preprocessed_data[phase].caption = self.preprocessed_data[phase].caption.str.lower()
            # preprocess
            captions_list = self.preprocessed_data[phase].caption.values.tolist()
            for i in range(len(captions_list)):
                caption = captions_list[i]
                # truncate long captions
                max_length = 18
                caption = caption.split(" ")
                if len(caption) > max_length:
                    caption = caption[:max_length]
                caption = " ".join(caption)
                # add start symbol
                caption = '<START> ' + caption
                # add end symbol
                caption += ' <END>'
                captions_list[i] = caption
                # filter out empty element
                caption = filter(None, caption)
            # replace with the new column
            new_captions = pandas.DataFrame({'caption': captions_list})
            self.preprocessed_data[phase].caption = new_captions.caption
            # sort the csv file by the lengths of descriptions
            self.preprocessed_data[phase] = self.preprocessed_data[phase].iloc[(-self.preprocessed_data[phase].caption.str.len()).argsort()].reset_index(drop=True)

        # build dict
        self._make_dict()

    # transform all words to their indices in the dictionary
    def _tranform(self):
        for phase in ["train", "val", "test"]:
            self.transformed_data[phase] = copy.deepcopy(self.preprocessed_data[phase])
            captions_list = self.transformed_data[phase].caption.values.tolist()
            for i in range(len(captions_list)):
                temp_list = []
                for text in captions_list[i].split(" "):
                    # filter out empty element
                    if text and text in self.dict_word2idx.keys():
                        temp_list.append(self.dict_word2idx[text])
                    elif text and text not in self.dict_word2idx.keys():
                        temp_list.append(self.dict_word2idx["<UNK>"])
                    captions_list[i] = temp_list
            # replace with the new column
            transformed_captions = pandas.DataFrame({'caption': captions_list})
            self.transformed_data[phase].caption = transformed_captions.caption
            # # sort the csv file by the lengths of descriptions
            # self.tranformed_csv = self.tranformed_csv.iloc[(-self.tranformed_csv.caption.str.len()).argsort()].reset_index(drop=True)
