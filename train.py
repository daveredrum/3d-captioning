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
from encoders import *
from decoders import *
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
    test_size = args.test_size
    epoch = args.epoch
    verbose = args.verbose
    lr = args.learning_rate
    batch_size = args.batch_size
    model_type = args.model_type
    weight_decay = args.weight_decay
    if args.attention == "true":
        args.attention = True
    elif args.attention == "false":
        args.attention = False
    attention = args.attention
    pretrained = args.pretrained
    if pretrained:
        model_name = pretrained
    else:
        model_name = "shallow"

    print("\n[settings]")
    print("GPU:", args.gpu)
    print("model_type:", args.model_type)
    print("pretrained:", args.pretrained)
    print("attention:", args.attention)
    print("train_size:", args.train_size)
    print("valid_size:", args.valid_size)
    print("test_size:", args.test_size)
    print("epoch:", args.epoch)
    print("verbose:", args.verbose)
    print("batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("weight_decay:", args.weight_decay)
    print()

    ###################################################################
    #                                                                 #
    #                                                                 #
    #                   training for encoder-decoder                  #
    #                                                                 #
    #                                                                 #
    ###################################################################

    # for 2d encoder
    if model_type == "2d":
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
        if pretrained == "resnet50":
            # prepare the dataloader
            train_ds = ImageCaptionDataset(
                root, 
                train_captions, 
                "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.png224.hdf5"
            )
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            valid_ds = ImageCaptionDataset(
                root, 
                valid_captions, 
                "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.png224.hdf5"
            )
            valid_dl = DataLoader(valid_ds, batch_size=batch_size)
            test_ds = ImageCaptionDataset(
                root, 
                test_captions, 
                "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.png224.hdf5"
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
            print("initializing encoder....")
            print()
            encoder = EncoderResnet50().cuda()
        else:
            # prepare the dataloader
            train_ds = ImageCaptionDataset(
                root, 
                train_captions, 
                "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.png.hdf5"
            )
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            valid_ds = ImageCaptionDataset(
                root, 
                valid_captions, 
                "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.png.hdf5"
            )
            valid_dl = DataLoader(valid_ds, batch_size=batch_size)
            test_ds = ImageCaptionDataset(
                root, 
                test_captions, 
                "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.png.hdf5"
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
            print("initializing encoder....")
            print()
            encoder = Encoder2D().cuda()

    # for 3d encoder   
    elif model_type == "3d":
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
        # prepare the dataloader
        train_ds = ShapeCaptionDataset(
            root, 
            train_captions, 
            database="/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.hdf5"
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        # valid_ds = ShapeCaptionDataset(root, valid_captions)
        valid_ds = ShapeCaptionDataset(
            root, 
            valid_captions,
            database="/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid.hdf5"
        )
        valid_dl = DataLoader(valid_ds, batch_size=batch_size)
        test_ds = ShapeCaptionDataset(
            root, 
            test_captions,
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

    # for coco
    elif model_type == "coco":
        # preprocessing
        print("preparing data....")
        print()
        coco = COCO(
            pandas.read_csv("/mnt/raid/davech2y/COCO_2014/preprocessed/coco_train2014.caption.csv"), 
            pandas.read_csv("/mnt/raid/davech2y/COCO_2014/preprocessed/coco_valid2014.caption.csv"),
            [train_size, valid_size]
        )
        # split data
        train_captions = coco.transformed_data['train']
        valid_captions = coco.transformed_data['valid']
        dict_idx2word = coco.dict_idx2word
        dict_word2idx = coco.dict_word2idx
        corpus = coco.corpus
        # prepare the dataloader
        if pretrained:
            train_ds = COCOCaptionDataset(
                root, 
                train_captions, 
                database="/mnt/raid/davech2y/COCO_2014/preprocessed/coco_train2014_224.hdf5"
            )
            valid_ds = COCOCaptionDataset(
                root, 
                valid_captions,
                database="/mnt/raid/davech2y/COCO_2014/preprocessed/coco_valid2014_224.hdf5"
            )
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            valid_dl = DataLoader(valid_ds, batch_size=batch_size)
            dataloader = {
                'train': train_dl,
                'valid': valid_dl
            }
            # initialize the encoder
            if pretrained == "resnet50":
                if attention:
                    print("initializing encoder: resnet50 with attention....")
                    print()
                    encoder = AttentionEncoderResnet50().cuda()
                else:
                    print("initializing encoder: resnet50....")
                    print()
                    encoder = EncoderResnet50().cuda()
            elif pretrained == "vgg16":
                if attention:
                    print("initializing encoder: vgg16 with attention....")
                    print()
                    encoder = AttentionEncoderVGG16().cuda()
                else:
                    print("initializing encoder: vgg16....")
                    print()
                    encoder = EncoderVGG16().cuda()
            elif pretrained == "vgg16_bn":
                if attention:
                    print("initializing encoder: vgg16_bn with attention....")
                    print()
                    encoder = AttentionEncoderVGG16BN().cuda()
                else:
                    print("initializing encoder: vgg16_bn....")
                    print()
                    encoder = EncoderVGG16BN().cuda()
            else:
                print("invalid model name, terminating...")
                return
        else:
            train_ds = COCOCaptionDataset(
                root, 
                train_captions, 
                database="/mnt/raid/davech2y/COCO_2014/preprocessed/coco_train2014.hdf5"
            )
            valid_ds = COCOCaptionDataset(
                root, 
                valid_captions,
                database="/mnt/raid/davech2y/COCO_2014/preprocessed/coco_valid2014.hdf5"
            )
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            valid_dl = DataLoader(valid_ds, batch_size=batch_size)
            dataloader = {
                'train': train_dl,
                'valid': valid_dl
            }
            # initialize the encoder
            print("initializing encoder....")
            print()
            encoder = Encoder2D().cuda()
        

    else:
        print("invalid model type, terminating.....")
        return

    # define the decoder
    input_size = dict_word2idx.__len__() + 1
    hidden_size = 512
    num_layer = 1
    if attention:
        if pretrained == "vgg16" or pretrained == "vgg16_bn":
            print("initializing decoder with attention....")
            decoder = AttentionDecoder2D(batch_size, input_size, hidden_size, 512, 14, num_layer).cuda()
        elif pretrained == "resnet50":
            print("initializing decoder with attention....")
            decoder = AttentionDecoder2D(batch_size, input_size, hidden_size, 2048, 7, num_layer).cuda()
    else:
        print("initializing decoder without attention....")        
        decoder = Decoder(input_size, hidden_size, num_layer).cuda()
    print("input_size:", input_size)
    print("dict_size:", dict_word2idx.__len__())
    print("hidden_size:", hidden_size)
    print("num_layer:", num_layer)
    print()


    # prepare the training parameters
    if pretrained:
        if attention:
            params = list(decoder.parameters())
        else:
            params = list(decoder.parameters()) + list(encoder.fc_layer.parameters())
    else:
        params = list(decoder.parameters()) + list(encoder.conv_layer.parameters()) + list(encoder.fc_layer.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    epoch = epoch
    verbose = verbose

    # training
    print("start training....")
    print()
    if attention:
        settings = "%s_%s_%s_ts%d_e%d_lr%f_wd%f_bs%d_vocal%d" % (model_type, model_name, "attention", train_size, epoch, lr, weight_decay, batch_size, input_size)
    else:
        settings = "%s_%s_%s_ts%d_e%d_lr%f_wd%f_bs%d_vocal%d" % (model_type, model_name, "noattention", train_size, epoch, lr, weight_decay, batch_size, input_size)
    encoder_decoder_solver = EncoderDecoderSolver(optimizer, criterion, model_type, settings)
    encoder_decoder_solver.train(encoder, decoder, dataloader, corpus, dict_word2idx, dict_idx2word, epoch, verbose, model_type, attention)

    # save
    print("save models...")
    torch.save(encoder, "models/encoder_%s.pth"  % settings)
    torch.save(decoder, "models/decoder_%s.pth"  % settings)

    # plot the result
    epochs = len(encoder_decoder_solver.log.keys())
    train_losses = [encoder_decoder_solver.log[i]["train_loss"] for i in range(epochs)]
    # valid_losses = [encoder_decoder_solver.log[i]["valid_loss"] for i in range(epochs)]train_perplexity
    train_perplexity = [encoder_decoder_solver.log[i]["train_perplexity"] for i in range(epochs)]
    train_blues_1 = [encoder_decoder_solver.log[i]["train_bleu_1"] for i in range(epochs)]
    train_blues_2 = [encoder_decoder_solver.log[i]["train_bleu_2"] for i in range(epochs)]
    train_blues_3 = [encoder_decoder_solver.log[i]["train_bleu_3"] for i in range(epochs)]
    train_blues_4 = [encoder_decoder_solver.log[i]["train_bleu_4"] for i in range(epochs)]
    valid_blues_1 = [encoder_decoder_solver.log[i]["valid_bleu_1"] for i in range(epochs)]
    valid_blues_2 = [encoder_decoder_solver.log[i]["valid_bleu_2"] for i in range(epochs)]
    valid_blues_3 = [encoder_decoder_solver.log[i]["valid_bleu_3"] for i in range(epochs)]
    valid_blues_4 = [encoder_decoder_solver.log[i]["valid_bleu_4"] for i in range(epochs)]
    train_cider = [encoder_decoder_solver.log[i]["train_cider"] for i in range(epochs)]
    valid_cider = [encoder_decoder_solver.log[i]["valid_cider"] for i in range(epochs)]
    # train_meteor = [encoder_decoder_solver.log[i]["train_meteor"] for i in range(epochs)]
    # valid_meteor = [encoder_decoder_solver.log[i]["valid_meteor"] for i in range(epochs)]
    train_rouge = [encoder_decoder_solver.log[i]["train_rouge"] for i in range(epochs)]
    valid_rouge = [encoder_decoder_solver.log[i]["valid_rouge"] for i in range(epochs)]

    # plot training curve
    print("plot training curves...")
    plt.switch_backend("agg")
    fig = plt.gcf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_losses, label="train_loss")
    # plt.plot(range(epochs), valid_losses, label="valid_loss")
    # plt.plot(range(epochs), train_perplexity, label="train_perplexity")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/training_curve_%s.png" % settings, bbox_inches="tight")
    # plot the bleu scores
    fig.clf()
    fig.set_size_inches(16,32)
    plt.subplot(4, 1, 1)
    plt.plot(range(epochs), train_blues_1, "C3", label="train_bleu")
    plt.plot(range(epochs), valid_blues_1, "C4", label="valid_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-1')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(range(epochs), train_blues_2, "C3", label="train_bleu")
    plt.plot(range(epochs), valid_blues_2, "C4", label="valid_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-2')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(range(epochs), train_blues_3, "C3", label="train_bleu")
    plt.plot(range(epochs), valid_blues_3, "C4", label="valid_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-3')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(range(epochs), train_blues_4, "C3", label="train_bleu")
    plt.plot(range(epochs), valid_blues_4, "C4", label="valid_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-4')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/bleu_curve_%s.png" % settings, bbox_inches="tight")
    # plot the cider scores
    fig.clf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_cider, label="train_cider")
    plt.plot(range(epochs), valid_cider, label="valid_cider")
    plt.xlabel('epoch')
    plt.ylabel('CIDEr')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/cider_curve_%s.png" % settings, bbox_inches="tight")
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
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("figs/rouge_curve_%s.png" % settings, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=100, help="train size for input captions")
    parser.add_argument("--valid_size", type=int, default=100, help="valid size for input captions")
    parser.add_argument("--test_size", type=int, default=100, help="test size for input captions")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="penalty on the optimizer")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--model_type", type=str, default="2d", help="type of model to train")
    parser.add_argument("--pretrained", type=str, default=None, help="vgg16/vgg16_bn/resnet50")
    parser.add_argument("--attention", type=str, default="false", help="true/false")
    args = parser.parse_args()
    main(args)
