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
from lib.data_embedding import *
from lib.configs import CONF
from model.encoder_attn import AdaptiveEncoder
from scipy.ndimage import zoom

def save_inters(model_id, weights, idx, path):
    for i in range(len(weights)):
        spatial_mask = weights[i][0].view(8, 8, 8).data.cpu().numpy()
        spatial_mask = zoom(spatial_mask, (8, 8, 8))
        spatial_mask = (spatial_mask * 255).astype(np.uint8)

        model_path = os.path.join(CONF.PATH.SHAPENET_ROOT.format(64), CONF.PATH.SHAPENET_NRRD.format(model_id[0], model_id[0]))
        readdata, _ = nrrd.read(model_path)

        inters = np.zeros((4, 64, 64, 64))
        inters[0][spatial_mask != 0] = readdata[0][spatial_mask != 0]
        inters[1][spatial_mask != 0] = readdata[1][spatial_mask != 0]
        inters[2][spatial_mask != 0] = readdata[2][spatial_mask != 0]
        inters[3][spatial_mask != 0] = readdata[3][spatial_mask != 0]
        inters = inters.astype(np.uint8)
        inters = np.swapaxes(inters, 1, 2)
        inters = np.swapaxes(inters, 1, 3)

        filename = os.path.join(path, "{}.nrrd".format(i))
        nrrd.write(filename, inters)

def extract(encoder, dataloader, root):
    model_count = {}
    for idx, (model_id, shape, text, _, _, _) in enumerate(dataloader):
        shape = shape.cuda()
        text = text.cuda()
        _, _, weights, attn_mask = encoder(shape, text)
        if model_id[0] in model_count.keys():
            model_count[model_id[0]] += 1
        else:
            model_count[model_id[0]] = 1

        # get attended part
        path = os.path.join(root, "attention", "{}-{}".format(model_id[0], model_count[model_id[0]]))
        if not os.path.exists(path):
            os.mkdir(path)
        save_inters(model_id, weights, idx, path)
        print("extracted and saved: {}-{}".format(model_id[0], model_count[model_id[0]]))

def main(args):
    # parse args
    root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path)
    voxel = int(args.path.split("_")[1][1:])
    encoder_path = os.path.join(root, "models/encoder.pth")
    
    phase = args.phase
    size = args.size
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
        1,
        False
    )
    dataset = ShapenetDataset(
        getattr(shapenet, "{}_data".format(phase)), 
        getattr(shapenet, "{}_idx2label".format(phase)), 
        getattr(shapenet, "{}_label2idx".format(phase)), 
        voxel,
        h5py.File(CONF.PATH.SHAPENET_DATABASE.format(voxel), "r")
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_shapenet)

    # report settings
    print("[settings]")
    print("extract attention masks from {} set".format(phase))
    print("size:", len(getattr(shapenet, "{}_data".format(phase))))
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    encoder = AdaptiveEncoder(shapenet.dict_idx2word.__len__(), args.path.split("_")[-1][8:]).cuda()
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    # feed and save as nrrd
    print("start extracting...\n")
    if not os.path.exists(os.path.join(root, "attention")):
        os.mkdir(os.path.join(root, "attention"))
    extract(encoder, dataloader, root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    parser.add_argument("--phase", type=str, default='val', help="train/val/test")
    parser.add_argument("--size", type=int, default=-1, help="size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)