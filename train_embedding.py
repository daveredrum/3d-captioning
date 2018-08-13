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
from lib.configs import CONF
from model.encoder_shape import ShapenetShapeEncoder
from model.encoder_text import ShapenetTextEncoder
from lib.losses import *
from lib.solver_embedding import *
import torch.multiprocessing as mp
import ctypes
import sys
from lib.utils import decode_log_embedding, draw_curves_embedding

def split_dataset(data, idx2label, size, voxel):
    for i in range(0, len(data), size):
        yield ShapenetDataset(data[i:i+size], idx2label, voxel)

def check_dataset(dataset, batch_size):
    flag = False
    for _, ds in dataset.items():
        if len(ds) % (batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL) != 0:
            flag = True
    
    return flag

def get_dataset(split_size, unique_batch_size, voxel, num_worker):
    # for training
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
        unique_batch_size,
        True
    )
    train_dataset = {
        x: y for x, y in zip(
            range(num_worker), 
            list(split_dataset(shapenet.train_data_agg, shapenet.train_idx2label, len(shapenet.train_data_agg) // num_worker, voxel))
        )
    }
    val_dataset = {
        x: y for x, y in zip(
            range(num_worker), 
            [ShapenetDataset(shapenet.val_data_agg, shapenet.val_idx2label, voxel) for _ in range(num_worker)]
        )
    }
    # for training
    eval_dataset = ShapenetDataset(
        getattr(shapenet, "{}_data".format(CONF.TRAIN.EVAL_DATASET)),
        getattr(shapenet, "{}_idx2label".format(CONF.TRAIN.EVAL_DATASET)), 
        voxel
    )

    return shapenet, train_dataset, val_dataset, eval_dataset


def get_dataloader(shapenet, train_dataset, val_dataset, eval_dataset, unique_batch_size, voxel, num_worker):
    # for training
    dataloader = {
        i: {
            'train': DataLoader(
                train_dataset[i], 
                batch_size=unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL,  
                collate_fn=collate_shapenet, 
                drop_last=check_dataset(train_dataset, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL)
            ),
            'val': DataLoader(
                val_dataset[i], 
                batch_size=unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL, 
                collate_fn=collate_shapenet, 
                drop_last=check_dataset(val_dataset, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL)
            )
        } for i in range(num_worker)
    }
    # for evaluation
    eval_dataset = ShapenetDataset(
        getattr(shapenet, "{}_data".format(CONF.TRAIN.EVAL_DATASET)),
        getattr(shapenet, "{}_idx2label".format(CONF.TRAIN.EVAL_DATASET)), 
        voxel
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=unique_batch_size, collate_fn=collate_shapenet)

    return dataloader, eval_dataloader


def main(args):
    # parse args
    voxel = args.voxel
    train_size = args.train_size
    val_size = args.val_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epoch = args.epoch
    unique_batch_size = args.batch_size
    num_worker = args.num_worker
    verbose = args.verbose
    gpu = args.gpu
    
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    mp.set_start_method('spawn', force=True)

    # prepare data
    print("\npreparing data...\n")
    shapenet, train_dataset, val_dataset, eval_dataset = get_dataset([train_size, val_size], unique_batch_size, voxel, num_worker)
    dataloader, eval_dataloader = get_dataloader(shapenet, train_dataset, val_dataset, eval_dataset, unique_batch_size, voxel, num_worker)
    train_per_worker = len(dataloader[0]['train']) * unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL
    val_per_worker = len(dataloader[0]['val']) * unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL
    
    # report settings
    print("[settings]")
    print("voxel:", voxel)
    print("train_size: {} samples -> {} pairs in total, {} pairs per worker".format(
        shapenet.train_size, 
        train_per_worker * num_worker, 
        train_per_worker
    ))
    print("val_size: {} samples -> {} pairs in total, {} pairs per worker".format(
        shapenet.val_size, 
        val_per_worker, 
        val_per_worker
    ))
    print("eval_size: {} samples -> evaluate on {} set".format(len(eval_dataset), CONF.TRAIN.EVAL_DATASET))
    print("learning_rate:", learning_rate)
    print("weight_decay:", weight_decay)
    print("epoch:", epoch)
    print("batch_size: {} shapes per batch, {} texts per batch".format(unique_batch_size, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL))
    print("num_worker:", num_worker)
    print("verbose:", verbose)
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    shape_encoder = ShapenetShapeEncoder().cuda()
    text_encoder = ShapenetTextEncoder(shapenet.dict_idx2word.__len__()).cuda()

    # initialize optimizer
    print("initializing optimizer...\n")
    criterion = {
        'walker': RoundTripLoss(weight=CONF.LBA.WALKER_WEIGHT),
        'visit': AssociationLoss(weight=CONF.LBA.VISIT_WEIGHT),
        'metric': InstanceMetricLoss(margin=CONF.ML.METRIC_MARGIN)
    }
    optimizer = torch.optim.Adam(list(shape_encoder.parameters()) + list(text_encoder.parameters()), lr=learning_rate, weight_decay=weight_decay)
    settings = CONF.TRAIN.SETTINGS.format("shapenet", voxel, shapenet.train_size, learning_rate, weight_decay, epoch, unique_batch_size, num_worker)
    if CONF.TRAIN.RANDOM_SAMPLE:
        settings += "_rand"
    solver = EmbeddingSolver(criterion, optimizer, settings)
    if not os.path.exists(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings)):
        os.mkdir(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings))

    # training
    print("start training...\n")
    shape_encoder.share_memory()
    text_encoder.share_memory()
    best = {
        'rank': mp.Value('i', 0),
        'epoch': mp.Value('i', 0),
        'total_loss': mp.Value(ctypes.c_float, float("inf")),
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
    return_log = mp.Queue()
    processes = []
    for rank in range(num_worker):
        p = mp.Process(target=solver.train, args=(shape_encoder, text_encoder, rank, best, lock, dataloader[rank], eval_dataloader, epoch, verbose, return_log))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    # report best
    print("------------------------[{}]best------------------------".format(best['rank'].value))
    print("[Loss] epoch: %d" % (
        best['epoch'].value
    ))
    print("[Loss] total_loss: %f" % (
        best['total_loss'].value
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
    print("[Loss] shape_norm_penalty: %f, text_norm_penalty: %f\n" % (
        best['shape_norm_penalty'].value,
        best['text_norm_penalty'].value
    ))

    # draw curves
    train_log, val_log = decode_log_embedding(return_log)
    draw_curves_embedding(train_log, val_log, os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings))


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
    parser.add_argument("--num_worker", type=int, default=1, help="number of workers") 
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)
