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

def get_dataset(data, idx2label, size, resolution):
    for i in range(0, len(data), size):
        yield ShapenetDataset(data[i:i+size], idx2label, resolution)

def check_dataset(dataset, batch_size):
    flag = False
    for _, ds in dataset.items():
        if len(ds) % batch_size <= 2:
            flag = True
    
    return flag

def get_dataloader(split_size, batch_size, resolution, num_worker):
    shapenet = Shapenet(
        [
            pickle.load(open("data/shapenet_split_train.p", 'rb')),
            pickle.load(open("data/shapenet_split_val.p", 'rb')),
            pickle.load(open("data/shapenet_split_test.p", 'rb'))
        ],
        [
            split_size[0],
            split_size[1],
            0
        ],
        batch_size,
        True
    )
    train_dataset = {
        x: y for x, y in zip(
            range(num_worker), 
            list(get_dataset(shapenet.train_data, shapenet.train_idx2label, len(shapenet.train_data) // num_worker, resolution))
        )
    }
    val_dataset = {
        x: y for x, y in zip(
            range(num_worker), 
            [ShapenetDataset(shapenet.val_data, shapenet.val_idx2label, resolution) for _ in range(num_worker)]
        )
    }
    dataloader = {
        i: {
            'train': DataLoader(
                train_dataset[i], 
                batch_size=batch_size,  
                collate_fn=collate_shapenet, 
                drop_last=check_dataset(train_dataset, batch_size)
            ),
            'val': DataLoader(
                val_dataset[i], 
                batch_size=batch_size, 
                collate_fn=collate_shapenet, 
                drop_last=check_dataset(val_dataset, batch_size)
            )
        } for i in range(num_worker)
    }
    

    return shapenet, dataloader


def main(args):
    # parse args
    voxel = args.voxel
    train_size = args.train_size
    val_size = args.val_size
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
    mp.set_start_method('spawn', force=True)

    # prepare data
    print("\npreparing data...\n")
    shapenet, dataloader = get_dataloader([train_size, val_size], batch_size * configs.N_CAPTION_PER_MODEL, voxel, num_worker)
    
    # report settings
    print("[settings]")
    print("voxel:", voxel)
    print("train_size: {} -> {}, {} per worker".format(shapenet.train_size, len(shapenet.train_data), len(shapenet.train_data) // num_worker))
    print("val_size: {} -> {}, {} per worker".format(shapenet.val_size, len(shapenet.val_data), len(shapenet.val_data)))
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
        'walker': RoundTripLoss(weight=configs.WALKER_WEIGHT),
        'visit': AssociationLoss(weight=configs.VISIT_WEIGHT),
        'metric': InstanceMetricLoss(margin=configs.METRIC_MARGIN)
    }
    optimizer = torch.optim.Adam(list(shape_encoder.parameters()) + list(text_encoder.parameters()), lr=learning_rate, weight_decay=weight_decay)
    settings = configs.SETTINGS.format("shapenet", voxel, shapenet.train_size, learning_rate, weight_decay, epoch, batch_size, num_worker)
    solver = EmbeddingSolver(criterion, optimizer, settings, configs.REDUCE_STEP)
    if not os.path.exists(os.path.join(configs.OUTPUT_EMBEDDING, settings)):
        os.mkdir(os.path.join(configs.OUTPUT_EMBEDDING, settings))

    # training
    print("start training...\n")
    shape_encoder.share_memory()
    text_encoder.share_memory()
    best = {
        'rank': mp.Value('i', 0),
        'epoch': mp.Value('i', 0),
        'loss': mp.Value(ctypes.c_float, float("inf")),
        'walker_loss_tst': mp.Value(ctypes.c_float, float("inf")),
        'walker_loss_sts': mp.Value(ctypes.c_float, float("inf")),
        'visit_loss_ts': mp.Value(ctypes.c_float, float("inf")),
        'visit_loss_st': mp.Value(ctypes.c_float, float("inf")),
        'metric_loss_st': mp.Value(ctypes.c_float, float("inf")),
        'metric_loss_tt': mp.Value(ctypes.c_float, float("inf")),
        'shape_norm_penalty': mp.Value(ctypes.c_float, float("inf")),
        'text_norm_penalty': mp.Value(ctypes.c_float, float("inf")),
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
    print("------------------------[{}]best------------------------".format(best['rank'].value))
    print("[Loss] epoch: %d" % (
        best['epoch'].value
    ))
    print("[Loss] val_loss: %f" % (
        best['loss'].value
    ))
    print("[Loss] walker_loss_tst: %f, walker_loss_sts: %f" % (
        best['walker_loss_tst'].value,
        best['walker_loss_sts'].value
    ))
    print("[Loss] visit_loss_ts: %f, visit_loss_st: %f" % (
        best['visit_loss_ts'].value,
        best['visit_loss_st'].value
    ))
    print("[Loss] metric_loss_st: %f, metric_loss_tt: %f" % (
        best['metric_loss_st'].value,
        best['metric_loss_tt'].value
    ))
    print("[Loss] shape_norm_penalty: %f, text_norm_penalty: %f" % (
        best['shape_norm_penalty'].value,
        best['text_norm_penalty'].value
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel", type=int, default=32, help="voxel resolution")
    parser.add_argument("--train_size", type=int, default=100, help="train size")
    parser.add_argument("--val_size", type=int, default=100, help="val size")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learepoch for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="penalty oepochimizer")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--num_worker", type=int, default=10, help="number of workers") 
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)
