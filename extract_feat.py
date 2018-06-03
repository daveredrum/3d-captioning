import h5py
import pandas as pd
import numpy as np
import torch
import encoders
import data
import time
import math
import argparse
from torch.utils.data import DataLoader

def main():
    model = encoders.AttentionVGG16BN().cuda()
    for phase in ["train", "valid"]:
        print()
        dataset = data.FeatureDataset(
            root="/mnt/raid/davech2y/COCO_2014/{}2014".format(phase),
            csv_file="/mnt/raid/davech2y/COCO_2014/preprocessed/coco_{}2014.caption.csv".format(phase)
        )
        dataloader = DataLoader(dataset, batch_size=50)
        database = h5py.File("/mnt/raid/davech2y/COCO_2014/preprocessed/{}_feature_vgg16.hdf5".format(phase), "w")
        storage = database.create_dataset("features", (len(dataset), 512 * 14 * 14), dtype="float")
        offset = 0
        for images in dataloader:
            start_since = time.time()
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
    main()
