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
from model.encoder_attn import AdaptiveEncoder
from lib.losses import *
from lib.solver_embedding import *
import torch.multiprocessing as mp
import ctypes
import sys
from lib.utils import decode_log_embedding, draw_curves_embedding

def check_dataset(dataset, batch_size):
    flag = False
    if len(dataset) % (batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL) != 0:
        flag = True
    
    return flag

def get_dataset(split_size, unique_batch_size, voxel):
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
    train_dataset = ShapenetDataset(
        shapenet.train_data_agg, 
        shapenet.train_idx2label, 
        shapenet.train_label2idx, 
        voxel, 
        h5py.File(CONF.PATH.SHAPENET_DATABASE.format(voxel), "r")
    )
    val_dataset = ShapenetDataset(
        shapenet.val_data_agg, 
        shapenet.val_idx2label, 
        shapenet.val_label2idx,
        voxel,
        h5py.File(CONF.PATH.SHAPENET_DATABASE.format(voxel), "r")
    )
    # for evaluation
    eval_dataset = ShapenetDataset(
        getattr(shapenet, "{}_data".format(CONF.TRAIN.EVAL_DATASET)),
        getattr(shapenet, "{}_idx2label".format(CONF.TRAIN.EVAL_DATASET)), 
        getattr(shapenet, "{}_label2idx".format(CONF.TRAIN.EVAL_DATASET)), 
        voxel,
        h5py.File(CONF.PATH.SHAPENET_DATABASE.format(voxel), "r")
    )

    return shapenet, train_dataset, val_dataset, eval_dataset


def get_dataloader(shapenet, train_dataset, val_dataset, eval_dataset, unique_batch_size, voxel):
    # for training
    dataloader = {
        'train': DataLoader(
            train_dataset, 
            batch_size=unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL,  
            collate_fn=collate_shapenet, 
            drop_last=check_dataset(train_dataset, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL)
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL, 
            collate_fn=collate_shapenet, 
            drop_last=check_dataset(val_dataset, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL)
        )
    }
    # for evaluation
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
    verbose = args.verbose
    gpu = args.gpu
    ver = args.ver
    if args.attention == 'adaptive':
        attention = True
        attention_type = 'adaptive'
    elif args.attention == 'false':
        attention = False
        attention_type = 'noattention'
        ver = None
    else:
        raise ValueError("invalid attention setting, terminating...")
    
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    mp.set_start_method('spawn', force=True)

    # prepare data
    print("\npreparing data...\n")
    shapenet, train_dataset, val_dataset, eval_dataset = get_dataset([train_size, val_size], unique_batch_size, voxel)
    dataloader, eval_dataloader = get_dataloader(shapenet, train_dataset, val_dataset, eval_dataset, unique_batch_size, voxel)
    train_per_worker = len(dataloader['train']) * unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL
    val_per_worker = len(dataloader['val']) * unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL
    
    # report settings
    print("[settings]")
    print("voxel:", voxel)
    print("train_size: {} samples -> {} pairs in total".format(
        shapenet.train_size, 
        train_per_worker
    ))
    print("val_size: {} samples -> {} pairs in total".format(
        shapenet.val_size, 
        val_per_worker
    ))
    print("eval_size: {} samples -> evaluate on {} set".format(len(eval_dataset), CONF.TRAIN.EVAL_DATASET))
    print("learning_rate:", learning_rate)
    print("weight_decay:", weight_decay)
    print("epoch:", epoch)
    print("batch_size: {} shapes per batch, {} texts per batch".format(unique_batch_size, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL))
    print("verbose:", verbose)
    print("gpu:", gpu)
    print("attention:", attention_type)
    print("version:", ver)

    # initialize models
    if attention:
        print("\ninitializing {} models...\n".format(attention_type))
        shape_encoder = AdaptiveEncoder(shapenet.dict_idx2word.__len__(), ver).cuda()
        text_encoder = None
    else:
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
    if attention:
        optimizer = torch.optim.Adam(shape_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(list(shape_encoder.parameters()) + list(text_encoder.parameters()), lr=learning_rate, weight_decay=weight_decay)
    settings = CONF.TRAIN.SETTINGS.format("shapenet", voxel, shapenet.train_size, learning_rate, weight_decay, epoch, unique_batch_size, attention_type + ver)
    if CONF.TRAIN.RANDOM_SAMPLE:
        settings += "_rand"
    solver = EmbeddingSolver(criterion, optimizer, settings)
    if not os.path.exists(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings)):
        os.mkdir(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings))

    # training
    print("start training...\n")
    best = {
        'epoch': 0,
        'total_loss': float("inf"),
        'walker_loss_tst': float("inf"),
        'walker_loss_sts': float("inf"),
        'visit_loss_ts': float("inf"),
        'visit_loss_st': float("inf"),
        'metric_loss_st': float("inf"),
        'metric_loss_tt': float("inf"),
        'shape_norm_penalty': float("inf"),
        'text_norm_penalty': float("inf"),
    }
    best, return_log = solver.train(shape_encoder, text_encoder, best, dataloader, eval_dataloader, epoch, verbose) 
    
    # report best
    print("------------------------best------------------------")
    print("[Loss] epoch: %d" % (
        best['epoch']
    ))
    print("[Loss] total_loss: %f" % (
        best['total_loss']
    ))
    print("[Loss] walker_loss_tst: %f, walker_loss_sts: %f" % (
        best['walker_loss_tst'],
        best['walker_loss_sts']
    ))
    print("[Loss] visit_loss_ts: %f, visit_loss_st: %f" % (
        best['visit_loss_ts'],
        best['visit_loss_st']
    ))
    print("[Loss] metric_loss_st: %f, metric_loss_tt: %f" % (
        best['metric_loss_st'],
        best['metric_loss_tt']
    ))
    print("[Loss] shape_norm_penalty: %f, text_norm_penalty: %f\n" % (
        best['shape_norm_penalty'],
        best['text_norm_penalty']
    ))

    # draw curves
    train_log, val_log = decode_log_embedding(return_log)
    draw_curves_embedding(train_log, val_log, os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel", type=int, default=64, help="voxel resolution")
    parser.add_argument("--train_size", type=int, default=100, help="train size")
    parser.add_argument("--val_size", type=int, default=100, help="val size")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learepoch for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="penalty oepochimizer")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    parser.add_argument("--attention", type=str, default='false', help="apply the attention: adaptive/false")
    parser.add_argument("--ver", type=str, default='2.1-c', help="1/2/2.1-a/2.1-b/2.1-c")
    args = parser.parse_args()
    main(args)
