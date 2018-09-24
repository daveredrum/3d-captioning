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
from PIL import Image
from lib.data_embedding import *
from lib.configs import CONF
from model.encoder_attn import SelfAttnShapeEncoder
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def apply_attn(model_id, raw_mask):
    spatial_mask = raw_mask.data.cpu().numpy()
    mask_size = spatial_mask.shape[0]
    up_scalar = int(64 / mask_size)
    spatial_mask = zoom(spatial_mask, (up_scalar, up_scalar, up_scalar))
    spatial_mask = (spatial_mask * 255).astype(np.uint8)

    model_path = os.path.join(CONF.PATH.SHAPENET_ROOT.format(64), CONF.PATH.SHAPENET_NRRD.format(model_id, model_id))
    readdata, _ = nrrd.read(model_path)

    applied = np.zeros((4, 64, 64, 64))
    applied[0][spatial_mask != 0] = readdata[0][spatial_mask != 0]
    applied[1][spatial_mask != 0] = readdata[1][spatial_mask != 0]
    applied[2][spatial_mask != 0] = readdata[2][spatial_mask != 0]
    applied[3][spatial_mask != 0] = readdata[3][spatial_mask != 0]
    applied = applied.astype(np.uint8)
    applied = np.swapaxes(applied, 1, 2)
    applied = np.swapaxes(applied, 1, 3)

    return applied

def filter_attn(model_ids, attn_masks):
    filtered = []
    for i in range(len(model_ids)):
        raw_mask_1 = attn_masks[0][1][i].view(8, 8, 8)
        raw_mask_2 = attn_masks[1][1][i].view(4, 4, 4)
        applied_1 = apply_attn(model_ids[i], raw_mask_1)
        applied_2 = apply_attn(model_ids[i], raw_mask_2)

        if applied_1.max() != 0 and applied_2.max() != 0:
            filtered.append(
                (
                    model_ids[i],
                    (
                        applied_1,
                        applied_2
                    )
                )
            )

    return filtered

def save_attn(encoder, dataloader, root):
    filtered_mask_list = []
    for model_ids, shapes, _, _, _, _ in dataloader:
        shapes = shapes.cuda()
        _, attn_masks = encoder(shapes)
        temp_list = filter_attn(model_ids, attn_masks)

        for (model_id, (mask_1, mask_2)) in temp_list:
            if not os.path.exists(os.path.join(root, "vis", "{}".format(model_id))):
                os.mkdir(os.path.join(root, "vis", "{}".format(model_id)))
            nrrd.write(os.path.join(root, "vis", "{}".format(model_id), "0.nrrd"), mask_1)
            nrrd.write(os.path.join(root, "vis", "{}".format(model_id), "1.nrrd"), mask_2)

        filtered_mask_list.extend(temp_list)

    return filtered_mask_list

def main(args):
    # parse args
    root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path)
    voxel = int(args.path.split("_")[1][1:])
    attn_type = args.path.split("_")[-1]
    encoder_path = os.path.join(root, "models/shape_encoder.pth")
    
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
    data = Embedding(
        [
            pickle.load(open("data/shapenet_split_train.p", 'rb')),
            pickle.load(open("data/shapenet_split_val.p", 'rb')),
            pickle.load(open("data/shapenet_split_test.p", 'rb'))
        ],
        size_split,
        1,
        False
    )
    dataset = EmbeddingDataset(
        getattr(data, "{}_data".format(phase)), 
        getattr(data, "{}_idx2label".format(phase)), 
        getattr(data, "{}_label2idx".format(phase)), 
        voxel,
        h5py.File(CONF.PATH.SHAPENET_DATABASE.format(voxel), "r"),
        aggr_shape=True
    )
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=collate_embedding)

    # report settings
    print("[settings]")
    print("extract attention masks from {} set".format(phase))
    print("size:", len(dataset))
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    encoder = SelfAttnShapeEncoder(attn_type, is_final=True).cuda()
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    # feed and save as nrrd
    print("start extracting...\n")
    if not os.path.exists(os.path.join(root, "vis")):
        os.mkdir(os.path.join(root, "vis"))
    filtered_mask_list = save_attn(encoder, dataloader, root)

    # report
    print("saved attention masks:", len(filtered_mask_list))
    print("abandoned attention masks:", len(dataset) - len(filtered_mask_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    parser.add_argument("--phase", type=str, default='val', help="train/val/test")
    parser.add_argument("--size", type=int, default=-1, help="size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)
