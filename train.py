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
    root = "/mnt/raid/davech2y/3d_captioning/ShapeNetCore_vol/nrrd_256_filter_div_128_solid/"
    captions = pandas.read_csv("captions.tablechair.csv")
    total_size = args.total_size
    split_ratio = args.split_ratio
    epoch = args.epoch
    verbose = args.verbose
    lr = args.learning_rate
    batch_size = args.batch_size
    # preprocessing
    captions = Caption(captions.iloc[:total_size])
    captions.preprocess()
    captions.tranform()
    # split data
    train_size = math.floor(total_size * (1 - split_ratio))
    train_captions = captions.tranformed_csv.iloc[:train_size]
    valid_captions = captions.tranformed_csv.iloc[train_size:total_size].reset_index(drop=True)

    ###################################################################
    #                                                                 #
    #                                                                 #
    #                   training for encoder-decoder                  #
    #                                                                 #
    #                                                                 #
    ###################################################################

    # prepare the dataloader
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    train_ds = PipelineDataset(root, train_captions, transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    valid_ds = PipelineDataset(root, valid_captions, transform)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    dataloader = {
        'train': train_dl,
        'valid': valid_dl
    }

    # # load the pretrained encoder
    # encoder = torch.load("data/encoder.pth").cuda()

    # initialize the encoder
    encoder = Encoder().cuda()

    # define the decoder
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
    encoder_decoder_solver = EncoderDecoderSolver(optimizer, criterion)
    encoder_decoder_solver.train(encoder, decoder, dataloader, epoch, verbose)

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
    plt.savefig("data/training_curve_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (total_size, epoch, lr, batch_size, input_size))
    torch.save(encoder, "data/encoder_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (total_size, epoch, lr, batch_size, input_size))
    torch.save(decoder, "data/decoder_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (total_size, epoch, lr, batch_size, input_size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_size", type=int, default=100, help="total size for input captions")
    parser.add_argument("--split_ratio", type=float, default=0.1, help="valid set ratio")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    args = parser.parse_args()
    print(args)
    print()
    print("start training....")
    print()
    main(args)
