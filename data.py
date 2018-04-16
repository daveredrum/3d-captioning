import torch
import pandas
import os
import re
import warnings
import operator
import copy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# pytorch dataset

class ShapeDataset(Dataset):
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
        self.dict_word2idx = {word_list[i][0]: i for i in range(len(word_list))}
        self.dict_idx2word = {i: word_list[i][0] for i in range(len(word_list))}
        

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
