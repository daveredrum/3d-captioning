import pandas 
import numpy as np
import os
import re
import operator
import math
import argparse
import torch
import torch.nn as nn
from data import *
from encoders import *
from decoders import *
import capeval.bleu.bleu as capbleu
import capeval.cider.cider as capcider
import capeval.meteor.meteor as capmeteor
import capeval.rouge.rouge as caprouge
from utils import *


class Report():
    def __init__(self, corpus, candidates, cider, num=3):
        '''
        params:
        - corpus: a dict containing image_ids and corresponding captions
        - candidates: a dict containing several dicts indexed by different beam sizes, in which images_ids
                      and corresponding generated captions are stored
        - cider: a dict containing several tuples indexed by different beam sizes, in which the mean CIDEr
                      scores and the per-image CIDEr scores are stored
        - num: number of images shown in the report, 3 by default
        '''
        self.corpus = corpus
        self.candidates = candidates
        self.cider = cider
        self.beam_sizes = list(candidates.keys())
        self.image_ids = list(corpus.keys())
        self.chosen = self._pick()

    def _pick(self):
        # return a dict of dicts containing images and captions
        chosen = {bs:None for bs in self.beam_sizes}
        for bs in self.beam_sizes:
            assert self.image_ids == list(self.candidates[bs].keys())
            pairs = [(image_id, score) for x, y in zip(self.image_ids, self.cider[bs][1])]
            # choose the images with the highest scores, picking the first caption in candidates
            highest = sorted(pairs, reverse=True, key=lambda x: x[1])[:3]
            highest = [(highest[i][0], self.candidates[bs][highest[i][0]][0]) for i in range(len(highest))]
            # the same thing for the lowest
            lowest = sorted(pairs, key=lambda x: x[1])[:3]
            lowest = [(lowest[i][0], self.candidates[bs][lowest[i][0]][0]) for i in range(len(lowest))]
            # choose the images with the closest scores to the mean scores
            med_pairs = [(image_id, abs(score - self.cider[bs][0])) for x, y in zip(self.image_ids, self.cider[bs][1])]
            med = sorted(med_pairs, key=lambda x: x[1])[:3]
            med = [(med[i][0], self.candidates[bs][med[i][0]][0]) for i in range(len(med))]
            # add into chosen
            chosen[bs] = {
                'high': highest,
                'low': lowest,
                'medium': med
            }
        
        return chosen


###################################################################
#                                                                 #
#                                                                 #
#                             evaluation                          #
#                                                                 #
#                                                                 #
###################################################################

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train_size = args.train_size
    test_size = args.test_size
    batch_size = args.batch_size
    pretrained = args.pretrained
    encoder_path = args.encoder
    decoder_path = args.decoder
    if args.attention == "true":
        attention = True
    elif args.attention == "false":
        attention = False
    print("\n[settings]")
    print("GPU:", args.gpu)
    print("train_size:", args.train_size)
    print("test_size:", args.test_size)
    print("batch_size:", args.batch_size)
    print("pretrained:", args.pretrained)
    print()
    print("preparing data...")
    print()
    coco = COCO(
        # for training
        pandas.read_csv("/mnt/raid/davech2y/COCO_2014/preprocessed/coco_train2014.caption.csv"), 
        pandas.read_csv("/mnt/raid/davech2y/COCO_2014/preprocessed/coco_val2014.caption.csv"),
        pandas.read_csv("/mnt/raid/davech2y/COCO_2014/preprocessed/coco_test2014.caption.csv"),
        [train_size, 0, test_size]
    )
    dict_idx2word = coco.dict_idx2word
    corpus = coco.corpus["test"]
    test_ds = COCOCaptionDataset(
        "/mnt/raid/davech2y/COCO_2014/preprocessed/test_index.json", 
        coco.transformed_data['test'], 
        database="data/test_feature_{}.hdf5".format(pretrained)
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    print("initializing models...")
    print()
    encoder = torch.load(os.path.join("models", encoder_path)).cuda()
    decoder = torch.load(os.path.join("models", decoder_path)).cuda()
    print("evaluating...")
    print()
    encoder.eval()
    decoder.eval()
    beam_size = [3, 5, 7]
    candidates = {i:{} for i in beam_size}
    outputs = {i:{} for i in beam_size}
    bleu = {i:{} for i in beam_size}
    cider = {i:{} for i in beam_size}
    rouge = {i:{} for i in beam_size}
    for _, (model_ids, visuals, captions, cap_lengths) in enumerate(test_dl):
        visual_inputs = Variable(visuals, requires_grad=False).cuda()
        caption_inputs = Variable(captions[:, :-1], requires_grad=False).cuda()
        cap_lengths = Variable(cap_lengths, requires_grad=False).cuda()
        visual_contexts = encoder(visual_inputs)
        max_length = int(cap_lengths[0].item()) + 10
        for bs in beam_size:
            if attention:
                outputs[bs] = decoder.beam_search(visual_contexts, caption_inputs, bs, max_length)
                outputs[bs] = decode_attention_outputs(outputs[bs], None, dict_idx2word, "val")
            else:
                outputs[bs] = decoder.beam_search(visual_contexts, bs, max_length)
                outputs[bs] = decode_outputs(outputs[bs], None, dict_idx2word, "val")
            for model_id, output in zip(model_ids, outputs[bs]):
                if model_id not in candidates[bs].keys():
                    candidates[bs][model_id] = [output]
                else:
                    candidates[bs][model_id].append(output)

    for bs in beam_size:
        # compute
        bleu[bs] = capbleu.Bleu(4).compute_score(corpus, candidates[bs])
        cider[bs] = capcider.Cider().compute_score(corpus, candidates[bs])
        rouge[bs] = caprouge.Rouge().compute_score(corpus, candidates[bs])
        # report
        print("----------------------Beam_size: {}-----------------------".format(bs))
        print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][0], max(bleu[bs][1][0]), min(bleu[bs][1][0])))
        print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][1], max(bleu[bs][1][1]), min(bleu[bs][1][1])))
        print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][2], max(bleu[bs][1][2]), min(bleu[bs][1][2])))
        print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[bs][0][3], max(bleu[bs][1][3]), min(bleu[bs][1][3])))
        print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[bs][0], max(cider[bs][1]), min(cider[bs][1])))
        print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[bs][0], max(rouge[bs][1]), min(rouge[bs][1])))
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=100, help="train size for input captions")
    parser.add_argument("--test_size", type=int, default=0, help="test size for input captions")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--gpu", type=str, help="specify the graphic card")
    parser.add_argument("--pretrained", type=str, default=None, help="vgg/resnet")
    parser.add_argument("--attention", type=str, default=None, help="true/false")
    parser.add_argument("--encoder", type=str, default=None, help="path to encoder")
    parser.add_argument("--decoder", type=str, default=None, help="path to decoder")
    args = parser.parse_args()
    main(args)