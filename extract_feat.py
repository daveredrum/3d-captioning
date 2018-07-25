import os
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
    model_path = os.path.join("outputs/models/embeddings", args.model_path)
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    batch_size = args.batch_size
    verbose = args.verbose
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
        dataset = ShapenetDataset(getattr(shapenet, "{}_data".format(phase)))
        dataloader[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_shapenet)

    # report settings
    print("[settings]")
    print("train_size:", len(shapenet.train_data))
    print("val_size:", len(shapenet.val_data))
    print("test_size:", len(shapenet.test_data))
    print("batch_size:", batch_size)
    print("verbose:", verbose)
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    shape_embedding = ShapenetEmbeddingEncoder(model_path).cuda()
    shape_embedding.eval()

    # set database
    print("creating database in 'shapenet_embedding.hdf5'...\n")
    with h5py.File(os.path.join("/mnt/raid/davech2y/ShapeNetCore_vol/", "shapenet_embedding.hdf5"), "w", libver="latest") as database:
        datatype = np.dtype([
            # ('model_id', h5py.special_dtype(vlen=str)),
            # ('model_cat', h5py.special_dtype(vlen=str)),
            # ('model_cap', h5py.special_dtype(vlen=str)),
            ('global_feat', np.float32, (128,)),
            ('area_feat', np.float32, (256 * 16 * 16 * 16,))
        ])

        # extract
        label2cat = {
            -1: 'table',
            1: 'chair'
        }
        for phase in ["train", "val", "test"]:
            print("extracting {} set...\n".format(phase))
            offset = 0
            dataset = database.create_dataset(phase, (len(getattr(shapenet, "{}_data".format(phase))),), dtype=datatype)
            for model_id, shape, caption, _, label in dataloader[phase]:
                # load
                shape = shape.cuda()

                # feed
                area_feat, global_feat = shape_embedding(shape)

                # append
                for i in range(len(model_id)):
                    print(i)
                    cap = " ".join([shapenet.dict_idx2word[str(idx.item())] for idx in caption[i] if idx.item() != 0])
                    data = (
                        # model_id[0], 
                        # label2cat[label[i].item()], 
                        # cap, 
                        global_feat[i].data.cpu().numpy().astype(np.float32),
                        area_feat[i].data.cpu().numpy().reshape((-1,)).astype(np.float32)
                    )

                    # print(data[0])
                    # print(data[1])
                    # print(data[2])
                    # print(data[3].shape)
                    # print(data[4].shape)
                    # return

                    # store
                    dataset[offset + i] = np.array(data, dtype=datatype)

                offset += len(model_id)
                print("extracted and stored: [{}/{}]".format(offset, len(getattr(shapenet, "{}_data".format(phase)))))
            
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="path to the pretrained encoder")
    parser.add_argument("--train_size", type=int, default=100, help="train size")
    parser.add_argument("--val_size", type=int, default=100, help="val size")
    parser.add_argument("--test_size", type=int, default=100, help="test size")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)