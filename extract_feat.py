import h5py
import pandas as pd
import numpy as np
import torch
import encoders
import data
import time
import math
import os
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader

def main(args):
    if not args.phases:
        phases = ["train", "valid"]
    else:
        phases = [args.phases]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("\ninitializing model...")
    print()
    if args.pretrained == "vgg16_bn":
        model = encoders.AttentionVGG16BN().cuda()
    elif args.pretrained == "resnet101":
        model = encoders.AttentionResNet101().cuda()
    for phase in phases:
        print(phase)
        print()
        print("preparing...")
        print()
        dataset = data.FeatureDataset(
            database="/mnt/raid/davech2y/COCO_2014/preprocessed/coco_{}2014_224.hdf5".format(phase)
        )
        dataloader = DataLoader(dataset, batch_size=32)
        if not os.path.exists("data/"):
            os.mkdir("data/")
        database = h5py.File("data/{}_feature_{}.hdf5".format(phase, args.pretrained), "w", libver='latest')
        if args.pretrained == "vgg16_bn":
            storage = database.create_dataset("features", (len(dataset), 512 * 14 * 14), dtype="float")
        elif args.pretrained == "resnet101":
            storage = database.create_dataset("features", (len(dataset), 2048 * 7 * 7), dtype="float")
        offset = 0
        print("extracting...")
        print()
        for images in dataloader:
            start_since = time.time()
            images = Variable(images).cuda()
            features = model(images)
            batch_size = features.size(0)
            for idx in range(batch_size):
                storage[offset + idx] = features[idx].view(-1).data.cpu().numpy()
            offset += batch_size
            exetime_s = time.time() - start_since
            eta_s = exetime_s * (len(dataset) - offset)
            eta_m = math.floor(eta_s / 60)
            print("preprocessed and stored: %d, ETA: %dm %ds" % (offset, eta_m, eta_s - eta_m * 60))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, help="vgg16_bn/resnet101")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--phases", type=str, default=None, help="train/valid")
    args = parser.parse_args()
    main(args)
