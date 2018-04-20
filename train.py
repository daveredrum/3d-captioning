import pandas 
import numpy as np
import os
import re
import operator
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from constants import *
from data import *
from models import *
from solver import *
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

def main():
    # settings
    root = "/mnt/raid/davech2y/3d_captioning/ShapeNetCore_vol/nrrd_256_filter_div_128_solid/"
    captions = pandas.read_csv("captions.tablechair.csv").iloc[:200]
    visual_contexts = np.load("data/visual_context.npy")
    # preprocessing
    captions = Caption(captions)
    captions.preprocess()
    captions.tranform()
    # split data
    train_captions = captions.tranformed_csv.iloc[:100]
    valid_captions = captions.tranformed_csv.iloc[100:200].reset_index(drop=True)

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
    train_dl = DataLoader(train_ds, batch_size=50)
    valid_ds = PipelineDataset(root, valid_captions, transform)
    valid_dl = DataLoader(valid_ds, batch_size=50)
    dataloader = {
        'train': train_dl,
        'valid': valid_dl
    }

    # load the pretrained encoder
    encoder = torch.load("data/encoder.pth").cuda()

    # define the decoder
    input_size = captions.dict_word2idx.__len__() + 1
    hidden_size = 512
    num_layer = 2
    decoder = Decoder(input_size, hidden_size, num_layer).cuda()

    # prepare the training parameters
    optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.fc_layer.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epoch = 1000
    verbose = 10

    # training
    encoder_decoder_solver = EncoderDecoderSolver(optimizer, criterion)
    encoder_decoder_solver.train(encoder, decoder, dataloader, epoch, verbose)

    # plot the result
    epochs = len(encoder_decoder_solver.log.keys())
    train_losses = [encoder_decoder_solver.log[i]["train_loss"] for i in range(epochs)]
    valid_losses = [encoder_decoder_solver.log[i]["valid_loss"] for i in range(epochs)]

    fig = plt.gcf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_losses, label="train_loss")
    plt.plot(range(epochs), valid_losses, label="valid_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(range(0, epochs + 1, verbose))
    plt.legend()
    
    # save
    fig.savefig("data/decoder_curve.png")

if __name__ == "__main__":
    print("start training....")
    print()
    main()