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
from model.encoder_shape import ShapeEncoder
from model.encoder_text import TextEncoder
from model.encoder_attn import *
from lib.losses import *
from lib.solver_embedding import EmbeddingSolver
import torch.multiprocessing as mp
import ctypes
import sys
from lib.utils import decode_log_embedding, draw_curves_embedding, report_best

def check_dataset(dataset, batch_size):
    flag = False
    if len(dataset) % batch_size != 0:
        flag = True
    
    return flag

def get_dataset(split_size, unique_batch_size, resolution):
    # for training
    embedding = Embedding(
        [
            pickle.load(open("data/{}_split_train.p".format(CONF.TRAIN.DATASET), 'rb')),
            pickle.load(open("data/{}_split_val.p".format(CONF.TRAIN.DATASET), 'rb')),
            pickle.load(open("data/{}_split_test.p".format(CONF.TRAIN.DATASET), 'rb'))
        ],
        [
            split_size[0],
            split_size[1],
            0
        ],
        unique_batch_size,
        True
    )
    if CONF.TRAIN.DATASET == 'shapenet':
        train_dataset = EmbeddingDataset(
            embedding.train_data_agg, 
            embedding.train_idx2label, 
            embedding.train_label2idx, 
            resolution, 
            h5py.File(CONF.PATH.SHAPENET_DATABASE.format(resolution), "r")
        )
        val_dataset = EmbeddingDataset(
            embedding.val_data_agg, 
            embedding.val_idx2label, 
            embedding.val_label2idx,
            resolution,
            h5py.File(CONF.PATH.SHAPENET_DATABASE.format(resolution), "r")
        )
        # for evaluation
        eval_dataset = EmbeddingDataset(
            getattr(embedding, "{}_data".format(CONF.TRAIN.EVAL_DATASET)),
            getattr(embedding, "{}_idx2label".format(CONF.TRAIN.EVAL_DATASET)), 
            getattr(embedding, "{}_label2idx".format(CONF.TRAIN.EVAL_DATASET)), 
            resolution,
            h5py.File(CONF.PATH.SHAPENET_DATABASE.format(resolution), "r")
        )
    elif CONF.TRAIN.DATASET == 'primitives':
        train_dataset = EmbeddingDataset(
            embedding.train_data_agg, 
            embedding.train_idx2label, 
            embedding.train_label2idx, 
            resolution
        )
        val_dataset = EmbeddingDataset(
            embedding.val_data_agg, 
            embedding.val_idx2label, 
            embedding.val_label2idx,
            resolution
        )
        # for evaluation
        eval_dataset = EmbeddingDataset(
            getattr(embedding, "{}_data".format(CONF.TRAIN.EVAL_DATASET)),
            getattr(embedding, "{}_idx2label".format(CONF.TRAIN.EVAL_DATASET)), 
            getattr(embedding, "{}_label2idx".format(CONF.TRAIN.EVAL_DATASET)), 
            resolution
        )

    return embedding, train_dataset, val_dataset, eval_dataset


def get_dataloader(embedding, train_dataset, val_dataset, eval_dataset, unique_batch_size, resolution):
    dataloader = {
        'train': DataLoader(
            train_dataset, 
            batch_size=unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL,  
            collate_fn=collate_embedding, 
            drop_last=check_dataset(train_dataset, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL)
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL, 
            collate_fn=collate_embedding, 
            drop_last=check_dataset(val_dataset, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL)
        ),
        'eval': DataLoader(
            eval_dataset, 
            batch_size=unique_batch_size, 
            collate_fn=collate_embedding
        )
    }

    return dataloader

def get_attention(args):
    if CONF.TRAIN.ATTN == 'noattention' or CONF.TRAIN.ATTN == 'text2shape':
        attention = False
    else:
        attention = True
    
    attention_type = CONF.TRAIN.ATTN

    return attention, attention_type

def get_models(attention_type, embedding):
    if attention_type == 'noattention' or attention_type == 'text2shape':
        print("\ninitializing naive models...\n")
        shape_encoder = ShapeEncoder().cuda()
        text_encoder = TextEncoder(embedding.dict_idx2word.__len__()).cuda()
    else:
        print("\ninitializing {} models...\n".format(attention_type))
        shape_encoder = SelfAttnShapeEncoder(attention_type).cuda()
        text_encoder = SelfAttnTextEncoder(embedding.dict_idx2word.__len__()).cuda()

    return shape_encoder, text_encoder

def get_optimizer(attention_type, shape_encoder, text_encoder, learning_rate, weight_decay):
    optimizer = torch.optim.Adam(list(shape_encoder.parameters()) + list(text_encoder.parameters()), lr=learning_rate, weight_decay=weight_decay)

    return optimizer

def get_settings(resolution, train_size, learning_rate, weight_decay, epoch, unique_batch_size, attention_type):
    settings = CONF.TRAIN.SETTINGS.format(CONF.TRAIN.DATASET, resolution, train_size, learning_rate, weight_decay, epoch, unique_batch_size, attention_type)
    
    return settings

def get_solver(embedding, criterion, optimizer, settings, unique_batch_size):
    solver = EmbeddingSolver(embedding, criterion, optimizer, settings, unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL)
    
    return solver

def save_logs(log, best, settings):
    if not os.path.exists(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings, "logs")):
        os.mkdir(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings, "logs"))
    pickle.dump(best, open(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings, "logs", "best.p"), 'wb'))
    pickle.dump(log, open(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings, "logs", "log.p"), 'wb'))

def main(args):
    # parse args
    resolution = CONF.TRAIN.RESOLUTION
    train_size = CONF.TRAIN.TRAIN_SIZE
    val_size = CONF.TRAIN.VAL_SIZE
    learning_rate = CONF.TRAIN.LEARNING_RATE
    weight_decay = CONF.TRAIN.WEIGHT_DECAY
    unique_batch_size = CONF.TRAIN.BATCH_SIZE
    verbose = CONF.TRAIN.VERBOSE
    epoch = args.epoch
    gpu = args.gpu
    attention, attention_type = get_attention(args)
    
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # prepare data
    print("\npreparing data...\n")
    embedding, train_dataset, val_dataset, eval_dataset = get_dataset([train_size, val_size], unique_batch_size, resolution)
    dataloader = get_dataloader(embedding, train_dataset, val_dataset, eval_dataset, unique_batch_size, resolution)
    train_per_worker = len(dataloader['train']) * unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL
    val_per_worker = len(dataloader['val']) * unique_batch_size * CONF.TRAIN.N_CAPTION_PER_MODEL
    
    # report settings
    print("[settings]")
    print("resolution:", resolution)
    print("train_size: {} samples -> {} pairs in total".format(
        embedding.train_size, 
        train_per_worker
    ))
    print("val_size: {} samples -> {} pairs in total".format(
        embedding.val_size, 
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

    # initialize models
    shape_encoder, text_encoder = get_models(attention_type, embedding)

    # initialize optimizer
    print("initializing optimizer...\n")
    criterion = {
        'walker': RoundTripLoss(weight=CONF.LBA.WALKER_WEIGHT),
        'visit': AssociationLoss(weight=CONF.LBA.VISIT_WEIGHT),
        'metric': InstanceMetricLoss(margin=CONF.ML.METRIC_MARGIN)
    }
    optimizer = get_optimizer(attention_type, shape_encoder, text_encoder, learning_rate, weight_decay)
    settings = get_settings(resolution, embedding.train_size, learning_rate, weight_decay, epoch, unique_batch_size, attention_type)
    solver = get_solver(embedding, criterion, optimizer, settings, unique_batch_size)
    
    # training
    print("start training...\n")
    if not os.path.exists(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings)):
        os.mkdir(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings))
    best = {
        'epoch': 0,
        'total_score': 0,
        'recall_1_t2s': 0,
        'recall_5_t2s': 0,
        'ndcg_5_t2s': 0,
        'recall_1_s2t': 0,
        'recall_5_s2t': 0,
        'ndcg_5_s2t': 0,
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
    best, log = solver.train(shape_encoder, text_encoder, best, dataloader, epoch, verbose) 
    
    # report best
    report_best(best)

    # save logs
    save_logs(log, best, settings)

    # draw curves
    train_log, val_log, eval_log = decode_log_embedding(log) 
    draw_curves_embedding(train_log, val_log, eval_log, os.path.join(CONF.PATH.OUTPUT_EMBEDDING, settings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)
