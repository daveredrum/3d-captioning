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
from model.encoder_shape import *

def main(args):
    # parse args
    shape_encoder = os.path.join("outputs/models/embeddings", args.shape_encoder)
    text_encoder = os.path.join("outputs/models/embeddings", args.text_encoder)
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
            pickle.load(open("pretrained/shapenet_split_train.p", 'rb')),
            pickle.load(open("pretrained/shapenet_split_val.p", 'rb')),
            pickle.load(open("pretrained/shapenet_split_test.p", 'rb'))
        ],
        [
            train_size,
            val_size,
            test_size
        ]
    )
    dataloader = {}
    for phase in ["train", "val", "test"]:
        dataset = ShapenetDataset(getattr(shapenet, "{}_data".format(phase)), getattr(shapenet, "{}_idx2label".format(phase)), configs.VOXEL)
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
    shape_encoder = torch.load(shape_encoder).cuda()
    text_encoder = torch.load(text_encoder).cuda()
    shape_encoder.eval()
    text_encoder.eval()

    # extract
    for phase in ["train", "val", "test"]:
        print("extracting {} set...\n".format(phase))
        data = [None] * len(getattr(shapenet, "{}_data".format(phase)))
        with open(configs.SHAPENET_EMBEDDING.format(phase), 'wb') as database:
            offset = 0
            total_iter = len(dataloader[phase])
            for iter_id, (model_id, shape, text, _, _) in enumerate(dataloader[phase]):
                start = time.time()
                # load
                shape = shape.cuda()
                text = text.cuda()

                # feed
                shape_embedding = shape_encoder(shape)
                text_embedding = text_encoder(text)

                # append
                for i in range(len(model_id)):
                    cap = " ".join([shapenet.dict_idx2word[str(idx.item())] for idx in text[i] if idx.item() != 0])
                    data[offset + i] = (
                        model_id[i], 
                        cap,
                        shape_embedding[i].data.cpu().numpy(),
                        text_embedding[i].data.cpu().numpy()
                    )

                # report
                offset += len(model_id)
                exe_s = time.time() - start
                eta_s = exe_s * (total_iter - (iter_id + 1))
                eta_m = math.floor(eta_s / 60)
                eta_s = math.floor(eta_s % 60)
                print("extracted: {}/{}, ETA: {}m {}s".format(offset, len(getattr(shapenet, "{}_data".format(phase))), eta_m, int(eta_s)))
            
            # store
            pickle.dump(data, database)
            print()
            
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_encoder", type=str, default=None, help="path to the pretrained shape encoder")
    parser.add_argument("--text_encoder", type=str, default=None, help="path to the pretrained text encoder")
    parser.add_argument("--train_size", type=int, default=100, help="train size")
    parser.add_argument("--val_size", type=int, default=100, help="val size")
    parser.add_argument("--test_size", type=int, default=100, help="test size")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)