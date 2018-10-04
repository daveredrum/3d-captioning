import os
import time
import math
import numpy as np
import h5py
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import zoom

# HACK
import sys
sys.path.append(".")
from lib.data_caption import *
from lib.configs import CONF
from model.decoders_caption import AttentionEncoderDecoder


def apply_attn(model_id, pairs):
    applied_mask_list = []
    raw_nrrd = nrrd.read(os.path.join(CONF.PATH.SHAPENET_ROOT.format(CONF.EVAL.RESOLUTION), CONF.PATH.SHAPENET_NRRD.format(model_id, model_id)))[0]
    for step_id in range(len(pairs)):
        upscaled_mask = F.upsample(pairs[step_id][2].view(1, 1, 4, 4, 4), scale_factor=16, mode="trilinear", align_corners=True).squeeze()

        attended = np.zeros((4, 64, 64, 64))
        attended[raw_nrrd != 0] = 255
        attended[:3] *= upscaled_mask.data.cpu().numpy()
        attended = attended.astype(np.uint8)

        attended = np.swapaxes(attended, 1, 2)
        attended = np.swapaxes(attended, 1, 3)

        applied_mask_list.append(
            (
                pairs[step_id][0],
                attended
            )
        )

        # alpha = CONF.EVAL.ALPHA
        # applied = attended * alpha + raw_nrrd * (1 - alpha)
        # applied = applied.astype(np.uint8)

        # applied_mask_list.append(
        #     (
        #         pairs[step_id][0],
        #         applied
        #     )
        # )

    return applied_mask_list


def save_attn(pipeline, dataloader, embeddings, root):
    applied_mask_list = []
    for i, (model_ids, _, _, embeddings_interm, _) in enumerate(dataloader):
        model_id = model_ids[0]
        inputs = embeddings_interm.cuda()
        pairs = pipeline.visual_attention(inputs, embeddings.dict_word2idx, embeddings.dict_idx2word)
        temp_list = apply_attn(model_id, pairs)

        for step_id, (token, applied_mask) in enumerate(temp_list):
            if not os.path.exists(os.path.join(root, "vis", "{}".format(model_id))):
                os.mkdir(os.path.join(root, "vis", "{}".format(model_id)))
            nrrd.write(os.path.join(root, "vis", "{}".format(model_id), "{}_[{}].nrrd".format(step_id, token)), applied_mask)
        
        print("saved attention mask for: {}, {} models left".format(model_id, len(dataloader) - i - 1))    
        applied_mask_list.extend(temp_list)

    return applied_mask_list

def main(args):
    # parse args
    embedding_path = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, "[FIN]shapenet_v64_trs11921_lr0.0002_wd0.0005_e20_bs100_selfnew-sep-cf/embedding/embedding.p")
    caption_path = os.path.join(CONF.PATH.OUTPUT_CAPTION, "[FIN]shapenet_selfnew-sep-cf_att2in_trs59777_vs7435_e50_lr0.00010_w0.00001_bs100_vocab3521_beam1")
    # attn_type = args.path.split("_")[-1]
    encoder_path = os.path.join(caption_path, "models/encoder.pth")
    decoder_path = os.path.join(caption_path, "models/decoder.pth")
    
    # phase = args.phase
    # size = args.size
    gpu = args.gpu

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # prepare data
    print("\npreparing data...\n")
    phase2idx = {'train': 0, 'val': 1, 'test': 2}
    size_split = [-1] * 3
    # size_split[phase2idx[CONF.EVAL.EVAL_DATASET]] = CONF.EVAL.NUM_CHOSEN
    embeddings = PretrainedEmbeddings(pickle.load(open(embedding_path, 'rb')), size_split)
    dataset = CaptionDataset(
        getattr(embeddings, "{}_text".format(CONF.EVAL.EVAL_DATASET)), 
        getattr(embeddings, "{}_shape".format(CONF.EVAL.EVAL_DATASET)),
        aggr_shape=True
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_ec)

    # report settings
    print("[settings]")
    print("extract attention masks from {} set".format(CONF.EVAL.EVAL_DATASET))
    print("size:", len(dataset))
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    pipeline = AttentionEncoderDecoder(encoder_path, decoder_path)

    # feed and save as nrrd
    print("start extracting...\n")
    if not os.path.exists(os.path.join(caption_path, "vis")):
        os.mkdir(os.path.join(caption_path, "vis"))
    applied_mask_list = save_attn(pipeline, dataloader, embeddings, caption_path)

    # report
    print("\nsaved {} attention masks for {} models".format(len(applied_mask_list), len(dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--embedding", type=str, default=None, help="path to the embedding root folder")
    # parser.add_argument("--caption", type=str, default=None, help="path to the caption root folder")
    # parser.add_argument("--phase", type=str, default='val', help="train/val/test")
    # parser.add_argument("--size", type=int, default=-1, help="size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)
