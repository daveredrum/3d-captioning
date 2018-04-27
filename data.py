import torch
import pandas
import os
import re
import warnings
import operator
import copy
import nrrd
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
    
# pipeline dataset for the encoder-decoder of image-caption
class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.image_paths = copy.deepcopy(csv_file.modelId.values.tolist())
        self.image_paths = [
            os.path.join(root_dir, model_name, model_name + '.png') 
            for model_name in self.image_paths
        ]
        self.caption_lists = copy.deepcopy(csv_file.description.values.tolist())
        self.csv_file = copy.deepcopy(csv_file)
        self.transform = transform
        self.data_pairs = self._build_data_pairs()

    def _build_data_pairs(self):
        # initialize data pairs: (image_path, caption, cap_length)
        data_pairs = [(
            self.image_paths[i],
            self.caption_lists[i],
            len(self.caption_lists[i])
        ) for i in range(self.__len__())]
        # # sort data pairs according to cap_length in descending order
        # data_pairs = sorted(data_pairs, key=lambda item: item[2], reverse=True)
        # pad caption with 0 if it's length is not maximum
        for index in range(1, len(data_pairs)):
            for i in range(len(data_pairs[0][1]) - len(data_pairs[index][1])):
                data_pairs[index][1].append(0)
        
        return data_pairs

    def __len__(self):
        return self.csv_file.id.count()

    def __getitem__(self, idx):
        # return (image_inputs, padded_caption, cap_length)
        image = Image.open(self.data_pairs[idx][0])
        image = np.array(image)[:, :, :3]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, self.data_pairs[idx][1], self.data_pairs[idx][2]

# pipeline dataset for the encoder-decoder of shape-caption
class ShapeCaptionDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.shape_paths = copy.deepcopy(csv_file.modelId.values.tolist())
        self.shape_paths = [
            os.path.join(root_dir, model_name, model_name + '.nrrd') 
            for model_name in self.shape_paths
        ]
        self.caption_lists = copy.deepcopy(csv_file.description.values.tolist())
        self.csv_file = copy.deepcopy(csv_file)
        self.data_pairs = self._build_data_pairs()

    def _build_data_pairs(self):
        # initialize data pairs: (image_path, caption, cap_length)
        data_pairs = [(
            self.shape_paths[i],
            self.caption_lists[i],
            len(self.caption_lists[i])
        ) for i in range(self.__len__())]
        # # sort data pairs according to cap_length in descending order
        # data_pairs = sorted(data_pairs, key=lambda item: item[2], reverse=True)
        # pad caption with 0 if it's length is not maximum
        for index in range(1, len(data_pairs)):
            for i in range(len(data_pairs[0][1]) - len(data_pairs[index][1])):
                data_pairs[index][1].append(0)
        
        return data_pairs

    def __len__(self):
        return self.csv_file.id.count()

    def __getitem__(self, idx):
        # return (image_inputs, padded_caption, cap_length)
        shape, _ = nrrd.read(self.data_pairs[idx][0])
        shape = np.array(shape)[:3, :, :, :]
        shape = (shape - shape.min()) / (shape.max() - shape.min())
        shape = torch.FloatTensor(shape)

        return shape, self.data_pairs[idx][1], self.data_pairs[idx][2]

# process csv file
class Caption(object):
    def __init__(self, csv_file):
        self.original_csv = csv_file
        self.preprocessed_csv = None
        self.tranformed_csv = None
        self.dict_word2idx = None
        self.dict_idx2word = None

    def preprocess(self):
        # suppress all warnings
        warnings.simplefilter('ignore')
        # drop items without captions
        self.preprocessed_csv = self.original_csv.loc[self.original_csv.description.notnull()].reset_index(drop=True)
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
            caption += ' <END>'
            captions_list[i] = caption
            # filter out empty element
            caption = filter(None, caption)
        # replace with the new column
        new_captions = pandas.DataFrame({'description': captions_list})
        self.preprocessed_csv.description = new_captions.description
        # build dict
        self._make_dict()

    def _make_dict(self):
        # output the dictionary of captions
        # indices of words are the rank of frequencies
        captions_list = self.preprocessed_csv.description.values.tolist()
        word_list = {}
        for text in captions_list:
            try:
                for word in re.split("[ ]", text):
                    if word:
                        if word in word_list.keys():
                            word_list[word] += 1
                        else:
                            word_list[word] = 1
            except Exception:
                pass
        word_list = sorted(word_list.items(), key=operator.itemgetter(1), reverse=True)
        # indexing starts at 1
        self.dict_word2idx = {word_list[i][0]: i+1 for i in range(len(word_list))}
        self.dict_idx2word = {i+1: word_list[i][0] for i in range(len(word_list))}
        

    def tranform(self):
        # transform all words to their indices in the dictionary
        self.tranformed_csv = copy.deepcopy(self.preprocessed_csv)
        captions_list = self.tranformed_csv.description.values.tolist()
        for i in range(len(captions_list)):
            temp_list = []
            try:
                for text in captions_list[i].split(" "):
                    # filter out empty element
                    if text:
                        temp_list.append(self.dict_word2idx[text])
                captions_list[i] = temp_list
            except Exception:
                pass
        # replace with the new column
        transformed_captions = pandas.DataFrame({'description': captions_list})
        self.tranformed_csv.description = transformed_captions.description
        # sort the csv file by the lengths of descriptions
        self.tranformed_csv = self.tranformed_csv.iloc[(-self.tranformed_csv.description.str.len()).argsort()].reset_index(drop=True)

    def sanity_check(self):
        # check if the transformation is reversable
        captions_list = self.preprocessed_csv.description.values.tolist()
        reverse_list = self.tranformed_csv.description.values.tolist()
        for i in range(len(reverse_list)):
            temp_string = ""
            for index in reverse_list[i]:
                temp_string += self.dict_idx2word[index]
                if index != reverse_list[i][-1]:
                    temp_string += " "
            reverse_list[i] = temp_string
        
        return reverse_list.sort() == captions_list.sort()
