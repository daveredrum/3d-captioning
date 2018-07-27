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
import torch.optim as optim
import matplotlib.pyplot as plt
from lib.data_embedding import *
import nrrd
import lib.configs as configs
from model.encoder_shape import ShapenetShapeEncoder
from model.encoder_text import ShapenetTextEncoder
from lib.losses import *
from lib.solver_embedding import *
import torch.multiprocessing as mp
import ctypes
import sys

mp.set_start_method('spawn', force=True)

def get_dataset(data, size, resolution):
    for i in range(0, len(data), size):
        yield ShapenetDataset(data[i:i+size], resolution)

def get_dataloader(train_size, batch_size, resolution, num_worker):
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
    dataset = {x: y for x, y in zip(range(num_worker), list(get_dataset(shapenet.train_data, len(shapenet.train_data) // num_worker, resolution)))}
    dataloader = {i: DataLoader(dataset[i], batch_size=batch_size, shuffle=True, collate_fn=collate_shapenet) for i in range(num_worker)}
    for _, ds in dataset.items():
        if len(ds) % batch_size == 1:
            sys.exit('invalid batch size, try a bigger or smaller one, terminating...')

    return shapenet, dataloader


def main(args):
    # parse args
    voxel = args.voxel
    train_size = args.train_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epoch = args.epoch
    batch_size = args.batch_size
    num_worker = args.num_worker
    verbose = args.verbose
    gpu = args.gpu
    
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # prepare data
    print("\npreparing data...\n")
    shapenet, dataloader = get_dataloader(train_size, batch_size, voxel, num_worker)
    
    # report settings
    print("[settings]")
    print("voxel:", voxel)
    print("train_size:", len(shapenet.train_data))
    print("learning_rate:", learning_rate)
    print("weight_decay:", weight_decay)
    print("epoch:", epoch)
    print("batch_size:", batch_size)
    print("num_worker:", num_worker)
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
    settings = "trs{}_lr{}_wd{}_e{}_bs{}_mp{}".format(train_size, learning_rate, weight_decay, epoch, batch_size, num_worker)
    solver = EmbeddingSolver(criterion, optimizer, settings, 3) 

    # training
    print("start training...\n")
    shape_encoder.share_memory()
    text_encoder.share_memory()
    best = {
        'train_loss': mp.Value(ctypes.c_float, float("inf")),
        'walker_loss_tst': mp.Value(ctypes.c_float, float("inf")),
        'walker_loss_sts': mp.Value(ctypes.c_float, float("inf")),
        'visit_loss_ts': mp.Value(ctypes.c_float, float("inf")),
        'visit_loss_st': mp.Value(ctypes.c_float, float("inf")),
        'metric_loss_ss': mp.Value(ctypes.c_float, float("inf")),
        'metric_loss_st': mp.Value(ctypes.c_float, float("inf")),
    }
    lock = mp.Lock()
    processes = []
    for rank in range(num_worker):
        p = mp.Process(target=solver.train, args=(shape_encoder, text_encoder, rank, best, lock, dataloader[rank], epoch, verbose))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    # report best
    print("------------------------[{}]best------------------------".format(rank))
    print("[Loss] train_loss: %f" % (
        best['train_loss'].value
    ))
    print("[Loss] walker_loss_tst: %f, walker_loss_sts: %f" % (
        best['walker_loss_tst'].value,
        best['walker_loss_sts'].value
    ))
    print("[Loss] visit_loss_ts: %f, visit_loss_st: %f" % (
        best['visit_loss_ts'].value,
        best['visit_loss_st'].value
    ))
    print("[Loss] metric_loss_ss: %f, metric_loss_st: %f\n" % (
        best['metric_loss_ss'].value,
        best['metric_loss_st'].value
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel", type=int, default=32, help="voxel resolution")
    parser.add_argument("--train_size", type=int, default=100, help="train size")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="penalty on the optimizer")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--num_worker", type=int, default=10, help="number of workers")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)