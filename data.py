import torch
import pandas
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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


