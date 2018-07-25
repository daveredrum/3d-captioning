import pandas 
import numpy as np
import os
import re
import operator
import math
import h5py
import json
import argparse
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.optim as optim
import matplotlib.pyplot as plt
from lib.data_embedding import *
import nrrd
import lib.configs as configs
from model.encoder_shape import ShapenetShapeEncoder
from model.encoder_text import ShapenetTextEncoder
from lib.losses import *
from lib.solver_embedding import *

def main(args):
    # parse args
    train_size = args.train_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epoch = args.epoch
    batch_size = args.batch_size
    verbose = args.verbose
    gpu = args.gpu
    
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # prepare data
    print("\npreparing data...\n")
    shapenet = Shapenet(
        [
            pickle.load(open("pretrained/shapenet_split_train.p", 'rb')),
            pickle.load(open("pretrained/shapenet_split_val.p", 'rb')),
            pickle.load(open("pretrained/shapenet_split_test.p", 'rb'))
        ],
        [
            train_size,
            0,
            0
        ]
    )
    dataset = ShapenetDataset(shapenet.train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_shapenet)
    
    # report settings
    print("[settings]")
    print("train_size:", len(shapenet.train_data))
    print("learning_rate:", learning_rate)
    print("weight_decay:", weight_decay)
    print("epoch:", epoch)
    print("batch_size:", batch_size)
    print("verbose:", verbose)
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    shape_encoder = ShapenetShapeEncoder().cuda()
    text_encoder = ShapenetTextEncoder(shapenet.dict_idx2word.__len__()).cuda()
    shape_encoder.train()
    text_encoder.train()

    # initialize optimizer
    print("initializing optimizer...\n")
    criterion = {
        'walker_tst': RoundTripLoss(weight=configs.WALKER_WEIGHT),
        'walker_sts': RoundTripLoss(weight=configs.WALKER_WEIGHT),
        'visit_ts': AssociationLoss(weight=configs.VISIT_WEIGHT),
        'visit_st': AssociationLoss(weight=configs.VISIT_WEIGHT),
        'metric_ss': MetricLoss(margin=configs.METRIC_MARGIN),
        'metric_st': MetricLoss(margin=configs.METRIC_MARGIN)
    }
    optimizer = torch.optim.Adam(list(shape_encoder.parameters()) + list(text_encoder.parameters()), lr=learning_rate, weight_decay=weight_decay)
    settings = "trs{}_lr{}_wd{}_e{}_bs{}".format(train_size, learning_rate, weight_decay, epoch, batch_size)
    solver = EmbeddingSolver(criterion, optimizer, settings, 3) 

    # training
    print("start training...\n")
    solver.train(shape_encoder, text_encoder, dataloader, epoch, verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=100, help="train size")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="penalty on the optimizer")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)