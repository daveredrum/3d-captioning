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
    test_captions = captions.transformed_data['test']
    dictionary = captions.dict_idx2word
    test_ds = ShapeCaptionDataset(
        root, 
        test_captions, 
        mode="hdf5", 
        database="/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.hdf5"
    )
    test_dl = DataLoader(test_ds, batch_size=1)

    # testing
    print("testing...")
    encoder_path = "models/encoder_%s_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (model_type, train_size, epoch, lr, batch_size, input_size)
    decoder_path = "models/decoder_%s_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (model_type, train_size, epoch, lr, batch_size, input_size)
    encoder_decoder = EncoderDecoder(encoder_path, decoder_path)

    descriptions = []
    images = []
    for i, (_, visual_inputs, _, _) in enumerate(test_dl):
        image_inputs = Variable(visual_inputs[0]).cuda()
        descriptions += encoder_decoder.generate_text(image_inputs, dictionary, 50)
        images.append(image_inputs)
        
    # edit the descriptions
    for i in range(len(descriptions)):
        text = descriptions[i].split(" ")
        new = []
        count = 0
        for j in range(len(text)):
            new.append(text[j])
            count += 1
            if count == 12:
                new.append("\n")
                count = 0
        descriptions[i] = " ".join(new)
    
    # plot testing results
    plt.switch_backend("agg")
    fig = plt.gcf()
    fig.set_size_inches(8, 4 * len(descriptions))
    fig.set_facecolor('white')
    for i in range(len(descriptions)):
        plt.subplot(len(descriptions), 1, i+1)
        plt.imshow(transforms.ToPILImage()(images[i].cpu().view(3, 64, 64)))
        plt.text(80, 32, descriptions[i], fontsize=14)
    plt.savefig("figs/testing_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size))

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