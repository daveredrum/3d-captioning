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
from model.encoder_attn import AdaptiveEncoder
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def summarize(root, tempo_attn, dict):
    for key, item in tempo_attn.items():
        mask, text = item
        fig = plt.figure()
        fig.set_size_inches(16, 20)

        # model image
        plt.subplot2grid((4, 6), (0, 0))
        model_id = key.split('-')[0]
        img = Image.open(os.path.join(CONF.PATH.SHAPENET_ROOT.format(64), CONF.PATH.SHAPENET_IMG.format(model_id, model_id)))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

        # attention distribution
        plt.subplot2grid((4, 6), (0, 1), colspan=5)
        cap = [dict[str(idx)] for idx in text[0]]
        plt.bar(range(20), mask, tick_label=cap, edgecolor='none')
        for i, spine in enumerate(plt.gca().spines.values()):
            if i == 2:
                continue
            spine.set_visible(False)
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)

        # attended parts
        for i in range(CONF.TRAIN.MAX_LENGTH + 1):
            plt.subplot2grid((4, 6), (i // 6 + 1, i % 6))
            img = Image.open(os.path.join(root, "attention", key, "{}.png".format(i)))
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.text(35, 95, cap[i], fontsize=12, bbox=dict(facecolor='#e1e1e1', edgecolor='#e1e1e1'))
            
        plt.savefig(os.path.join(root, "vis", "{}.png".format(key)), bbox_inches="tight")
        fig.clf()

# def render(root):
#     assert os.path.exists(os.path.join(root, "attention"))
#     file_list = os.listdir(os.path.join(root, "attention"))
#     cline = "ssc/render-voxels.js --input {}/{}.nrrd"
#     for file_name in file_list:
#         for i in range(CONF.TRAIN.MAX_LENGTH):
#             os.system(cline.format(root + '/' + "attention" + '/' + file_name, i))


def save_inters(model_id, weights, idx, path):
    for i in range(len(weights)):
        spatial_mask = weights[i][0][:512].view(8, 8, 8).data.cpu().numpy()
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
    tempo_attn = {}
    model_count = {}
    for idx, (model_id, shape, text, _, _, _) in enumerate(dataloader):
        shape = shape.cuda()
        text = text.cuda()
        _, text, weights, attn_mask = encoder(shape, text)
        if model_id[0] in model_count.keys():
            model_count[model_id[0]] += 1
        else:
            model_count[model_id[0]] = 1
        tempo_attn["{}-{}".format(model_id[0], model_count[model_id[0]])] = (attn_mask[0][0].data.cpu().numpy(), text.data.cpu().numpy())

        # get attended part
        path = os.path.join(root, "attention", "{}-{}".format(model_id[0], model_count[model_id[0]]))
        if not os.path.exists(path):
            os.mkdir(path)
            save_inters(model_id, weights, idx, path)
            print("extracted and saved: {}-{}".format(model_id[0], model_count[model_id[0]]))
        else:
            print("exists and skipped: {}-{}".format(model_id[0], model_count[model_id[0]]))

    return tempo_attn


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
    tempo_attn = extract(encoder, dataloader, root)

    # dump
    print("save attention mask...")
    pickle.dump(tempo_attn, open(os.path.join(root, "attention", "attn_mask.p"), 'wb'))

    # # render nrrd
    # print("start rendering...\n")
    # render(root)

    # # plot
    # print("start generating summaries...")
    # if not os.path.exists(os.path.join(root, "vis")):
    #     os.mkdir(os.path.join(root, "vis"))
    # summarize(root, tempo_attn, shapenet.dict_idx2word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    parser.add_argument("--phase", type=str, default='val', help="train/val/test")
    parser.add_argument("--size", type=int, default=-1, help="size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)
