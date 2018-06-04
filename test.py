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
    mode = args.mode
    database = args.database
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    train_csv = pandas.read_csv(args.train_csv)
    val_csv = pandas.read_csv(args.val_csv)
    outname = args.outname
    encoder_path = args.encoder
    decoder_path = args.decoder
    phase = args.phase
    if mode == "2d":
        # preprocessing
        print("preparing data....")
        print()
        captions = Caption(train_csv, [train_size, val_size, test_size])
        # split data
        transformed_csv = captions.transformed_data[phase]
        dictionary = captions.dict_idx2word
        dataset = ImageCaptionDataset(None, transformed_csv, database)
        dataloader = DataLoader(dataset, batch_size=1)

    elif mode == "3d":
        # preprocessing
        print("preparing data....")
        print()
        captions = Caption(train_csv, [train_size, val_size, test_size])
        # split data
        transformed_csv = captions.transformed_data[phase]
        dictionary = captions.dict_idx2word
        dataset = ShapeCaptionDataset(None, transformed_csv, database)
        dataloader = DataLoader(dataset, batch_size=1)
    
    elif mode == "coco":
        # preprocessing
        print("preparing data....")
        print()
        captions = COCO(train_csv, val_csv, [train_size, val_size])
        # split data
        transformed_csv = captions.transformed_data[phase]
        dictionary = captions.dict_idx2word
        dataset = COCOCaptionDataset(None, transformed_csv, database)
        dataloader = DataLoader(dataset, batch_size=1)

    # testing
    print("initializing encoder and decoder...")
    print()
    encoder_decoder = EncoderDecoder(encoder_path, decoder_path)

    # captioning
    print("testing...")
    print()
    descriptions = []
    images = []
    for i, (model_id, visual_inputs, _, _) in enumerate(dataloader):
        if mode == "coco":
            inputs = Variable(visual_inputs).cuda() # image
            images.append(model_id)
        elif mode == "2d":
            inputs = Variable(visual_inputs).cuda() # image
            images.append(visual_inputs)
        elif mode == "3d":
            inputs = Variable(visual_inputs[1]).cuda() # shape
            images.append(visual_inputs[0][0])
        descriptions += encoder_decoder.generate_text(inputs, dictionary, 50)
        # only use part of the dataset
        if i >= 10:
            break
            
            
    # edit the descriptions
    for i in range(len(descriptions)):
        text = descriptions[i].split(" ")
        new = []
        count = 0
        for j in range(len(text)):
            new.append(text[j])
            count += 1
            if count == 16:
                new.append("\n")
                count = 0
        descriptions[i] = " ".join(new)

    # plot testing results
    print("saving results...")
    plt.switch_backend("agg")

    fig = plt.gcf()
    fig.set_size_inches(8, 4 * len(descriptions))
    fig.set_facecolor('white')
    for i in range(len(descriptions)):
        plt.subplot(len(descriptions), 1, i+1)
        if mode == "coco":
            image_path = transformed_csv.file_name.loc[transformed_csv.image_id == int(images[i][0])].drop_duplicates().iloc[0]
            image = Image.open(os.path.join("/mnt/raid/davech2y/COCO_2014/%s2014" % phase, image_path)).resize((64, 64))
            plt.imshow(image)
        elif mode == "2d":
            image = images[i].numpy()
            image = transforms.ToPILImage(image)
            plt.imshow(image)
        elif mode == "3d":
            plt.imshow(Image.open(images[i]).resize((64, 64)))
        plt.text(80, 32, descriptions[i], fontsize=12)
    # fig.tight_layout()
    plt.savefig("results/%s.png" % outname, bbox_inches="tight")
    fig.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="coco", help="source and type of the input data")
    parser.add_argument("--phase", type=str, help="train or val")
    parser.add_argument("--database", type=str, default=None, help="path to the preprocessed data")
    parser.add_argument("--train_csv", type=str, default=None, help="csv file for the training captions")
    parser.add_argument("--val_csv", type=str, default=None, help="csv file for the valation captions")
    parser.add_argument("--train_size", type=int, default=100, help="train size for input captions")
    parser.add_argument("--val_size", type=int, default=100, help="val size for input captions")
    parser.add_argument("--test_size", type=int, default=100, help="test size for input captions")
    parser.add_argument("--encoder", type=str, default=None, help="path to the encoder")
    parser.add_argument("--decoder", type=str, default=None, help="path to the decoder")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--outname", type=str, help="output name for the results")
    args = parser.parse_args()
    main(args)