import pandas 
import numpy as np
import os
import re
import operator
import math
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from constants import *
from data import *
from decoders import *
from solver import *
import matplotlib
import matplotlib.pyplot as plt

def main(args):
    # settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    mode = args.mode
    phase = args.phase
    index = args.index
    outname = args.outname
    encoder_path = "/home/davech2y/3d_captioning/models/{}".format(args.encoder)
    decoder_path = "/home/davech2y/3d_captioning/models/{}".format(args.decoder)
    print("\n[settings]")
    print("mode:", mode)
    print("phase:", phase)
    print("index:", index)
    print("outname:", outname)
    print("encoder_path:", encoder_path)
    print("decoder_path:", decoder_path)
    print()
    
    # preprocessing
    print("preparing data....")
    print()
    model_name = encoder_path.split(".")[0]
    settings = model_name.split("_")
    if settings[4] == "attention":
        print("invalid model, terminating...")
        return
    captions = COCO(
        pandas.read_csv("/mnt/raid/davech2y/COCO_2014/preprocessed/coco_train2014.caption.csv"), 
        pandas.read_csv("/mnt/raid/davech2y/COCO_2014/preprocessed/coco_valid2014.caption.csv"), 
        [int(settings[6][2:]), math.floor(int(settings[6][2:]) / 10)]
    )
    # split data
    transformed_csv = captions.transformed_data[phase]
    dict_word2idx = captions.dict_word2idx
    dict_idx2word = captions.dict_idx2word
    dataset = COCOCaptionDataset(None, transformed_csv, "/mnt/raid/davech2y/COCO_2014/preprocessed/coco_{}2014_224.hdf5".format(phase))
    dataloader = DataLoader(dataset, batch_size=1)

    # testing
    print("initializing encoder and decoder...")
    print()
    encoder_decoder = AttentionEncoderDecoder(encoder_path, decoder_path)

    # testing
    print("generating captions...")
    print()
    for i, (model_id, visual_inputs, _, _) in enumerate(dataloader):
        if i == index:
            visual_inputs = Variable(visual_inputs).cuda()
            images = [model_id]
            descriptions = " ".join(encoder_decoder.generate_text(visual_inputs, dict_word2idx, dict_idx2word, 20))
            new = []
            for i, text in enumerate(descriptions.split(" ")):
                new.append(text)
                if i != 0 and i % 16 == 0:
                    new.append("\n")
            descriptions = [" ".join(new)]
            pairs = encoder_decoder.visual_attention(visual_inputs, dict_word2idx, dict_idx2word, 20)
            break
    descriptions += [pairs[i][0] for i in range(len(pairs))]
    images += [pairs[i][1] for i in range(len(pairs))]

    # plot testing results
    if not os.path.exists("/home/davech2y/3d_captioning/results/{}".format(outname)):
        os.mkdir("/home/davech2y/3d_captioning/results/{}".format(outname))
    print("saving results...")
    df = {i: pairs[i][2].view(-1).data.cpu().numpy().tolist() for i in range(len(images) - 1)}
    df = pandas.DataFrame(df)
    df.to_csv("results/{}/{}.csv".format(outname, outname), index=False)
    plt.switch_backend("agg")

    fig = plt.gcf()
    fig.set_size_inches(8, 4 * len(descriptions))
    fig.set_facecolor('white')
    for i in range(len(descriptions)):
        plt.subplot(len(descriptions), 1, i+1)
        if i == 0:
            image_path = transformed_csv.file_name.loc[transformed_csv.image_id == int(images[i][0])].drop_duplicates().iloc[0]
            image = Image.open(os.path.join("/mnt/raid/davech2y/COCO_2014/%s2014" % phase, image_path)).resize((64, 64))
            plt.imshow(image)
            plt.text(80, 32, descriptions[i], fontsize=12)
        else:
            plt.imshow(images[i].data.cpu().numpy())
            plt.text(80, 32, descriptions[i], fontsize=12)
    # fig.tight_layout()
    plt.savefig("results/{}/{}.png".format(outname, outname), bbox_inches="tight")
    fig.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="coco", help="source and type of the input data")
    parser.add_argument("--phase", type=str, help="train/valid")
    parser.add_argument("--index", type=int, default=0, help="index of the testing image")
    parser.add_argument("--encoder", type=str, default=None, help="path to the encoder")
    parser.add_argument("--decoder", type=str, default=None, help="path to the decoder")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--outname", type=str, help="output name for the results")
    args = parser.parse_args()
    main(args)
