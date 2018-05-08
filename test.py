import pandas 
import numpy as np
import os
import re
import operator
import math
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from constants import *
from data import *
from models import *
from solver import *
import matplotlib
import matplotlib.pyplot as plt

def main(args):
    # settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    root = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_128_solid/"
    captions = pandas.read_csv("captions.tablechair.csv")
    train_size = args.train_size
    valid_size = args.valid_size
    test_size = args.valid_size
    epoch = args.epoch
    lr = args.learning_rate
    batch_size = args.batch_size
    model_type = args.model_type
    # preprocessing
    print("preparing data....")
    captions = Caption(pandas.read_csv("captions.tablechair.csv"), [train_size, valid_size, test_size])
    input_size = captions.dict_word2idx.__len__() + 1
    # split data
    train_captions = captions.transformed_data['train']
    test_captions = captions.transformed_data['test']
    dictionary = captions.dict_idx2word
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    if model_type == "2d":
        train_ds = ImageCaptionDataset(root, train_captions, transform)
        test_ds = ImageCaptionDataset(root, test_captions, transform)
    elif model_type == "3d":
        train_ds = ShapeCaptionDataset(
            root, 
            train_captions,
            database="/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.hdf5"
        )
        test_ds = ShapeCaptionDataset(
            root, 
            test_captions, 
            database="/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.hdf5"
        )
    train_dl = DataLoader(train_ds, batch_size=1)
    test_dl = DataLoader(test_ds, batch_size=1)
    dataloader = {
        'train': train_dl,
        'test': test_dl
    }

    # testing
    print("testing...")
    encoder_path = "models/encoder_%s_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (model_type, train_size, epoch, lr, batch_size, input_size)
    decoder_path = "models/decoder_%s_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (model_type, train_size, epoch, lr, batch_size, input_size)
    encoder_decoder = EncoderDecoder(encoder_path, decoder_path)

    # captioning
    descriptions = {
        'train': [],
        'test': []
    }
    images = {
        'train': [],
        'test': []
    }
    for phase in ["train", "test"]:
        for i, (_, visual_inputs, _, _) in enumerate(dataloader[phase]):
            if model_type == "2d":
                inputs = Variable(visual_inputs).cuda() # image
                images[phase].append(visual_inputs)
            elif model_type == "3d":
                inputs = Variable(visual_inputs[1]).cuda() # shape
                images[phase].append(visual_inputs[0][0])
            descriptions[phase] += encoder_decoder.generate_text(inputs, dictionary, 50)
            # only use part of the dataset
            if i >= 10:
                break
            
            
        # edit the descriptions
        for i in range(len(descriptions)):
            text = descriptions[phase][i].split(" ")
            new = []
            count = 0
            for j in range(len(text)):
                new.append(text[j])
                count += 1
                if count == 16:
                    new.append("\n")
                    count = 0
            descriptions[phase][i] = " ".join(new)
    
        # plot testing results
        plt.switch_backend("agg")

        fig = plt.gcf()
        fig.set_size_inches(8, 4 * len(descriptions[phase]))
        fig.set_facecolor('white')
        for i in range(len(descriptions[phase])):
            plt.subplot(len(descriptions[phase]), 1, i+1)
            if model_type == "2d":
                plt.imshow(transforms.ToPILImage()(images[phase][i].view(3, 64, 64)))
            elif model_type == "3d":
                plt.imshow(Image.open(images[phase][i]).resize((64, 64)))
            plt.text(80, 32, descriptions[phase][i], fontsize=12)
        # fig.tight_layout()
        plt.savefig("figs/%s_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (phase, model_type, train_size, epoch, lr, batch_size, input_size), bbox_inches="tight")
        fig.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=100, help="train size for input captions")
    parser.add_argument("--valid_size", type=int, default=100, help="valid size for input captions")
    parser.add_argument("--test_size", type=int, default=100, help="test size for input captions")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--model_type", type=str, default="2d", help="type of model to train")
    args = parser.parse_args()
    print(args)
    print()
    main(args)