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
    verbose = args.verbose
    lr = args.learning_rate
    batch_size = args.batch_size
    model_type = args.model_type
    # preprocessing
    print("preparing data....")
    print()
    captions = Caption(pandas.read_csv("captions.tablechair.csv"), [train_size, valid_size, test_size])
    # split data
    train_captions = captions.transformed_data['train']
    valid_captions = captions.transformed_data['valid']
    dictionary = captions.dict_idx2word
    corpus = captions.corpus

    ###################################################################
    #                                                                 #
    #                                                                 #
    #                   training for encoder-decoder                  #
    #                                                                 #
    #                                                                 #
    ###################################################################

    # for 2d encoder
    if model_type == "2d":
        # prepare the dataloader
        transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
        train_ds = ImageCaptionDataset(root, train_captions, transform)
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        valid_ds = ImageCaptionDataset(root, valid_captions, transform)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size)
        dataloader = {
            'train': train_dl,
            'valid': valid_dl
        }

        # # load the pretrained encoder
        # encoder = torch.load("data/encoder.pth").cuda()

        # initialize the encoder
        encoder = Encoder2D().cuda()

    # for 3d encoder   
    elif model_type == "3d":
        train_ds = ShapeCaptionDataset(root, train_captions)
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        valid_ds = ShapeCaptionDataset(root, valid_captions)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size)
        dataloader = {
            'train': train_dl,
            'valid': valid_dl
        }

        # # load the pretrained encoder
        # encoder = torch.load("data/encoder.pth").cuda()

        # initialize the encoder
        encoder = Encoder3D().cuda()

    else:
        print("invalid model type, exiting.....")
        return

    # define the decoder
    print("initializing models....")
    print()
    input_size = captions.dict_word2idx.__len__() + 1
    hidden_size = 512
    num_layer = 2
    decoder = Decoder(input_size, hidden_size, num_layer).cuda()

    # prepare the training parameters
    # optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.fc_layer.parameters()), lr=0.001)
    params = list(decoder.parameters()) + list(encoder.conv_layer.parameters()) + list(encoder.fc_layer.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    epoch = epoch
    verbose = verbose

    # training
    print("start training....")
    print()
    encoder_decoder_solver = EncoderDecoderSolver(optimizer, criterion, model_type)
    encoder_decoder_solver.train(encoder, decoder, dataloader, corpus, dictionary, epoch, verbose)

    # plot the result
    epochs = len(encoder_decoder_solver.log.keys())
    train_losses = [encoder_decoder_solver.log[i]["train_loss"] for i in range(epochs)]
    valid_losses = [encoder_decoder_solver.log[i]["valid_loss"] for i in range(epochs)]

    plt.switch_backend("agg")
    fig = plt.gcf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_losses, label="train_loss")
    plt.plot(range(epochs), valid_losses, label="valid_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
   
    # save
    plt.savefig("figs/training_curve_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size))
    torch.save(encoder, "models/encoder_%s_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (model_type, train_size, epoch, lr, batch_size, input_size))
    torch.save(decoder, "models/decoder_%s_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (model_type, train_size, epoch, lr, batch_size, input_size))

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
