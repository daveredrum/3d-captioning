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
    root = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid/"
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
    test_captions = captions.transformed_data['test']
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
        test_ds = ImageCaptionDataset(root, test_captions, transform)
        test_dl = DataLoader(test_ds, batch_size=batch_size)
        dataloader = {
            'train': train_dl,
            'valid': valid_dl,
            'test': test_dl
        }

        # # load the pretrained encoder
        # encoder = torch.load("data/encoder.pth").cuda()

        # initialize the encoder
        encoder = Encoder2D().cuda()

    # for 3d encoder   
    elif model_type == "3d":
        train_ds = ShapeCaptionDataset(
            root, 
            train_captions, 
            mode="hdf5", 
            database="/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.hdf5"
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        # valid_ds = ShapeCaptionDataset(root, valid_captions)
        valid_ds = ShapeCaptionDataset(
            root, 
            valid_captions,
            mode="hdf5", 
            database="/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.hdf5"
        )
        valid_dl = DataLoader(valid_ds, batch_size=batch_size)
        test_ds = ShapeCaptionDataset(
            root, 
            test_captions,
            mode="hdf5", 
            database="/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.hdf5"
        )
        test_dl = DataLoader(test_ds, batch_size=1)
        dataloader = {
            'train': train_dl,
            'valid': valid_dl,
            'test': test_dl
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
    encoder_decoder_solver.train(encoder, decoder, dataloader, corpus, dictionary, epoch, verbose, model_type)

    # save
    print("save models...")
    torch.save(encoder, "models/encoder_%s_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (model_type, train_size, epoch, lr, batch_size, input_size))
    torch.save(decoder, "models/decoder_%s_ts%d_e%d_lr%f_bs%d_vocal%d.pth"  % (model_type, train_size, epoch, lr, batch_size, input_size))

    # plot the result
    epochs = len(encoder_decoder_solver.log.keys())
    train_losses = [encoder_decoder_solver.log[i]["train_loss"] for i in range(epochs)]
    valid_losses = [encoder_decoder_solver.log[i]["valid_loss"] for i in range(epochs)]
    train_blues_1 = [encoder_decoder_solver.log[i]["train_blue_1"] for i in range(epoch)]
    train_blues_2 = [encoder_decoder_solver.log[i]["train_blue_2"] for i in range(epoch)]
    train_blues_3 = [encoder_decoder_solver.log[i]["train_blue_3"] for i in range(epoch)]
    train_blues_4 = [encoder_decoder_solver.log[i]["train_blue_4"] for i in range(epoch)]
    valid_blues_1 = [encoder_decoder_solver.log[i]["valid_blue_1"] for i in range(epoch)]
    valid_blues_2 = [encoder_decoder_solver.log[i]["valid_blue_2"] for i in range(epoch)]
    valid_blues_3 = [encoder_decoder_solver.log[i]["valid_blue_3"] for i in range(epoch)]
    valid_blues_4 = [encoder_decoder_solver.log[i]["valid_blue_4"] for i in range(epoch)]
    train_cider = [encoder_decoder_solver.log[i]["train_cider"] for i in range(epoch)]
    valid_cider = [encoder_decoder_solver.log[i]["valid_cider"] for i in range(epoch)]
    # train_meteor = [encoder_decoder_solver.log[i]["train_meteor"] for i in range(epoch)]
    # valid_meteor = [encoder_decoder_solver.log[i]["valid_meteor"] for i in range(epoch)]
    train_rouge = [encoder_decoder_solver.log[i]["train_rouge"] for i in range(epoch)]
    valid_rouge = [encoder_decoder_solver.log[i]["valid_rouge"] for i in range(epoch)]

    # plot training curve
    print("plot training curves...")
    plt.switch_backend("agg")
    fig = plt.gcf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_losses, label="train_loss")
    plt.plot(range(epochs), valid_losses, label="valid_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/training_curve_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size), bbox_inches="tight")
    # plot the bleu scores
    fig.clf()
    fig.set_size_inches(16,32)
    plt.subplot(4, 1, 1)
    plt.plot(range(epochs), train_blues_1, "C3", label="train_blue")
    plt.plot(range(epochs), valid_blues_1, "C4", label="valid_blue")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-1')
    plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(range(epochs), train_blues_2, "C3", label="train_blue")
    plt.plot(range(epochs), valid_blues_2, "C4", label="valid_blue")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-2')
    plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(range(epochs), train_blues_3, "C3", label="train_blue")
    plt.plot(range(epochs), valid_blues_3, "C4", label="valid_blue")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-3')
    plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(range(epochs), train_blues_4, "C3", label="train_blue")
    plt.plot(range(epochs), valid_blues_4, "C4", label="valid_blue")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-4')
    plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/bleu_curve_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size), bbox_inches="tight")
    # plot the cider scores
    fig.clf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_cider, label="train_cider")
    plt.plot(range(epochs), valid_cider, label="valid_cider")
    plt.xlabel('epoch')
    plt.ylabel('CIDEr')
    plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/cider_curve_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size), bbox_inches="tight")
    # # plot the meteor scores
    # fig.clf()
    # fig.set_size_inches(16,8)
    # plt.plot(range(epochs), train_meteor, label="train_meteor")
    # plt.plot(range(epochs), valid_meteor, label="valid_meteor")
    # plt.xlabel('epoch')
    # plt.ylabel('METEOR')
    # plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    # plt.legend()
    # plt.savefig("figs/meteor_curve_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size), bbox_inches="tight")
    # plot the rouge scores
    fig.clf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_rouge, label="train_rouge")
    plt.plot(range(epochs), valid_rouge, label="valid_rouge")
    plt.xlabel('epoch')
    plt.ylabel('ROUGE_L')
    plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/rouge_curve_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size), bbox_inches="tight")


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
