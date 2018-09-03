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
from model.encoder_attn import AdaptiveEncoder



def extract(shape_encoder, text_encoder, dataloader, shapenet, phase, verbose=False):
    data = {}
    offset = 0
    total_iter = len(dataloader)
    for iter_id, (model_id, shape, text, _, _, _) in enumerate(dataloader):
        start = time.time()
        # load
        shape = shape.cuda()
        text = text.cuda()

        # non-attentive
        if text_encoder:
            shape_embedding = shape_encoder(shape)
            text_embedding = text_encoder(text)
        else:
            shape_embedding, text_embedding , _, _ = shape_encoder(shape, text)


        # dump
        for i in range(len(model_id)):
            if shapenet:
                cap = " ".join([shapenet.dict_idx2word[str(idx.item())] for idx in text[i] if idx.item() != 0])
            else:
                cap = None
            if model_id[i] in data.keys():
                data[model_id[i]]['text_embedding'].append(
                    (
                        cap,
                        text_embedding[i].data.cpu().numpy()
                    )
                ) 
            else:
                data[model_id[i]] = {
                    'shape_embedding': shape_embedding[i].data.cpu().numpy(),
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

    # # aggregate shape embeddings
    # for key in data.keys():
    #     data[key]['shape_embedding'] = np.mean(data[key]['shape_embedding'], axis=0)

    return data

def main(args):
    # parse args
    root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path)
    voxel = int(args.path.split("_")[1][1:])
    if args.path.split("_")[-1] == "noattention":
        shape_encoder_path = os.path.join(root, "models/shape_encoder.pth")
        text_encoder_path = os.path.join(root, "models/text_encoder.pth")
    else:
        shape_encoder_path = os.path.join(root, "models/encoder.pth")
        text_encoder_path = None
    
    phase = args.phase
    size = args.size
    batch_size = args.batch_size
    gpu = args.gpu

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # prepare data
    print("\npreparing data...\n")
    phase2idx = {'train': 0, 'val': 1, 'test': 2}
    size_split = [-1] * 3
    size_split[phase2idx[phase]] = size
    shapenet = Shapenet(
        [
            pickle.load(open("data/shapenet_split_train.p", 'rb')),
            pickle.load(open("data/shapenet_split_val.p", 'rb')),
            pickle.load(open("data/shapenet_split_test.p", 'rb'))
        ],
        size_split,
        batch_size,
        False
    )
    dataset = ShapenetDataset(
        getattr(shapenet, "{}_data".format(phase)), 
        getattr(shapenet, "{}_idx2label".format(phase)), 
        getattr(shapenet, "{}_label2idx".format(phase)), 
        voxel,
        h5py.File(CONF.PATH.SHAPENET_DATABASE.format(voxel), "r")
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_shapenet)

    # report settings
    print("[settings]")
    print("phase:", phase)
    print("size:", len(getattr(shapenet, "{}_data".format(phase))))
    print("batch_size:", batch_size)
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    if text_encoder_path:
        shape_encoder = ShapenetShapeEncoder().cuda()
        shape_encoder.load_state_dict(torch.load(shape_encoder_path))
        shape_encoder.eval()
        text_encoder = ShapenetTextEncoder(shapenet.dict_idx2word.__len__()).cuda()
        text_encoder.load_state_dict(torch.load(text_encoder_path))
        text_encoder.eval()
    else:
        shape_encoder = AdaptiveEncoder(shapenet.dict_idx2word.__len__(), args.path.split("_")[-1][8:]).cuda()
        shape_encoder.load_state_dict(torch.load(shape_encoder_path))
        shape_encoder.eval()
        text_encoder = None

    # extract
    if not os.path.exists(os.path.join(root, "embeddings")):
        os.mkdir(os.path.join(root, "embeddings"))
    
    print("extracting {} set...\n".format(phase))
    with open(os.path.join(root, "embeddings", "{}.p".format(phase)), 'wb') as database:
        # extract
        data = extract(shape_encoder, text_encoder, dataloader, shapenet, phase, True)
        
        # store
        pickle.dump(data, database)
        print()
            
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    parser.add_argument("--phase", type=str, default='val', help="train/val/test")
    parser.add_argument("--size", type=int, default=100, help="train size")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)
