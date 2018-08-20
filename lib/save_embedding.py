import os
import time
import math
import numpy as np
import h5py
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# HACK
import sys
sys.path.append(".")
from lib.configs import CONF
from lib.data_embedding import *
from model.encoder_shape import *
from model.encoder_text import *

def extract(shape_encoder, text_encoder, dataloader, shapenet, phase, verbose=False):
    data = {}
    offset = 0
    total_iter = len(dataloader)
    for iter_id, (model_id, shape, text, _, _) in enumerate(dataloader):
        start = time.time()
        # load
        shape = shape.cuda()
        text = text.cuda()

        # feed
        if text_encoder:
            shape_embedding = shape_encoder(shape)
            text_embedding = text_encoder(text)
        else:
            shape_embedding, text_embedding = shape_encoder(shape, text)

        # append
        for i in range(len(model_id)):
            if shapenet:
                cap = " ".join([shapenet.dict_idx2word[str(idx.item())] for idx in text[i] if idx.item() != 0])
            else:
                cap = None
            if model_id[i] in data.keys():
                data[model_id[i]]['shape_embedding'].append(shape_embedding[i].data.cpu().numpy())
                data[model_id[i]]['text_embedding'].append(
                    (
                        cap,
                        text_embedding[i].data.cpu().numpy()
                    )
                ) 
            else:
                data[model_id[i]] = {
                    'shape_embedding': [shape_embedding[i].data.cpu().numpy()],
                    'text_embedding': [
                        (
                            cap,
                            text_embedding[i].data.cpu().numpy()
                        )
                    ]
                }

        # report
        if verbose and shapenet:
            offset += len(model_id)
            exe_s = time.time() - start
            eta_s = exe_s * (total_iter - (iter_id + 1))
            eta_m = math.floor(eta_s / 60)
            eta_s = math.floor(eta_s % 60)
            print("extracted: {}/{}, ETA: {}m {}s".format(offset, len(getattr(shapenet, "{}_data".format(phase))), eta_m, int(eta_s)))

    # aggregate shape embeddings
    for key in data.keys():
        data[key]['shape_embedding'] = np.mean(data[key]['shape_embedding'], axis=0)

    return data

def main(args):
    # parse args
    root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path)
    voxel = args.voxel
    shape_encoder_path = os.path.join(root, "models/shape_encoder.pth")
    text_encoder_path = os.path.join(root, "models/text_encoder.pth")
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    batch_size = args.batch_size
    gpu = args.gpu

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # prepare data
    print("\npreparing data...\n")
    shapenet = Shapenet(
        [
            pickle.load(open("data/shapenet_split_train.p", 'rb')),
            pickle.load(open("data/shapenet_split_val.p", 'rb')),
            pickle.load(open("data/shapenet_split_test.p", 'rb'))
        ],
        [
            train_size,
            val_size,
            test_size
        ],
        batch_size,
        False
    )
    dataloader = {}
    for phase in ["train", "val", "test"]:
        dataset = ShapenetDataset(getattr(shapenet, "{}_data".format(phase)), getattr(shapenet, "{}_idx2label".format(phase)), voxel)
        dataloader[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_shapenet)

    # report settings
    print("[settings]")
    print("train_size:", len(shapenet.train_data))
    print("val_size:", len(shapenet.val_data))
    print("test_size:", len(shapenet.test_data))
    print("batch_size:", batch_size)
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    shape_encoder = ShapenetShapeEncoder().cuda()
    shape_encoder.load_state_dict(torch.load(shape_encoder_path))
    text_encoder = ShapenetTextEncoder(shapenet.dict_idx2word.__len__()).cuda()
    text_encoder.load_state_dict(torch.load(text_encoder_path))
    shape_encoder.eval()
    text_encoder.eval()

    # extract
    if not os.path.exists(os.path.join(root, "embeddings")):
        os.mkdir(os.path.join(root, "embeddings"))
    for phase in ["train", "val", "test"]:
        print("extracting {} set...\n".format(phase))
        with open(os.path.join(root, "embeddings", "{}.p".format(phase)), 'wb') as database:
            # extract
            data = extract(shape_encoder, text_encoder, dataloader[phase], shapenet, phase, True)
            
            # store
            pickle.dump(data, database)
            print()
            
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    parser.add_argument("--voxel", type=int, default=32, help="voxel resolution")
    parser.add_argument("--train_size", type=int, default=100, help="train size")
    parser.add_argument("--val_size", type=int, default=100, help="val size")
    parser.add_argument("--test_size", type=int, default=100, help="test size")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)