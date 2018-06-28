import pandas 
import numpy as np
import os
import re
import operator
import math
import argparse
import torch
import torch.nn as nn
from lib.pretrained_data import *
from model.encoders import *
from model.decoders import *
import lib.capeval.bleu.bleu as capbleu
import lib.capeval.cider.cider as capcider
import lib.capeval.meteor.meteor as capmeteor
import lib.capeval.rouge.rouge as caprouge
from lib.utils import *
import lib.configs as configs
import matplotlib.pyplot as plt


class Report():
    def __init__(self, corpus, candidates, cider, mode, num=3):
        '''
        params:
        - corpus: a dict containing image_ids and corresponding captions
        - candidates: a dict containing several dicts indexed by different beam sizes, in which images_ids
                      and corresponding generated captions are stored
        - cider: a dict containing several tuples indexed by different beam sizes, in which the mean CIDEr
                      scores and the per-image CIDEr scores are stored
        - mode: shapenet/primitive
        - num: number of images shown in the report, 3 by default
        '''
        self.corpus = corpus
        self.candidates = candidates
        self.cider = cider
        self.num = num
        self.mode = mode
        self.beam_sizes = list(candidates.keys())
        self.image_ids = list(corpus.keys())
        self.chosen = self._pick()

    def _pick(self):
        # return a dict of dicts containing images and captions
        chosen = {bs:None for bs in self.beam_sizes}
        for bs in self.beam_sizes:
            assert set(self.image_ids) == set(self.candidates[bs].keys())
            if bs == self.beam_sizes[0]:
                pairs = [(x, y) for x, y in zip(self.image_ids, self.cider[bs][1])]
                # choose the images with the highest scores, picking the first caption in candidates
                highest = sorted(pairs, reverse=True, key=lambda x: x[1])[:self.num]
                highest = [(highest[i][0], self.candidates[bs][highest[i][0]][0], self.corpus[highest[i][0]][0]) for i in range(len(highest))]
                # the same thing for the lowest
                lowest = sorted(pairs, key=lambda x: x[1])[:self.num]
                lowest = [(lowest[i][0], self.candidates[bs][lowest[i][0]][0], self.corpus[lowest[i][0]][0]) for i in range(len(lowest))]
                # choose the images with the closest scores to the mean scores
                med_pairs = [(x, abs(y - self.cider[bs][0])) for x, y in zip(self.image_ids, self.cider[bs][1])]
                med = sorted(med_pairs, key=lambda x: x[1])[:self.num]
                med = [(med[i][0], self.candidates[bs][med[i][0]][0], self.corpus[med[i][0]][0]) for i in range(len(med))]
                # add into chosen
                chosen[bs] = {
                    'high': sorted(highest, key=lambda x: x[0]),
                    'low': sorted(lowest, key=lambda x: x[0]),
                    'medium': sorted(med, key=lambda x: x[0])
                }
                cache = {
                    'high': [item[0] for item in chosen[bs]['high']],
                    'low': [item[0] for item in chosen[bs]['low']],
                    'medium': [item[0] for item in chosen[bs]['medium']]
                }
            else:
                chosen[bs] = {
                    'high': [(item, self.candidates[bs][item][0], self.corpus[item][0]) for item in cache['high']],
                    'low': [(item, self.candidates[bs][item][0], self.corpus[item][0]) for item in cache['low']],
                    'medium': [(item, self.candidates[bs][item][0], self.corpus[item][0]) for item in cache['medium']]
                }
                

        
        return chosen

    def __call__(self, path):
        plt.switch_backend("agg")
        for q in ["high", "low", "medium"]:
            fig = plt.figure(figsize=(8, 24), dpi=100, facecolor='w')
            fig.clf()
            for i in range(self.num):
                image_id = self.chosen['1'][q][i][0]
                plt.subplot(self.num, 1, i+1)
                if self.mode == 'shapenet':
                    plt.imshow(Image.open(os.path.join(configs.SHAPE_ROOT, "{}/{}.png".format(image_id, image_id))).convert('RGB').resize((224, 224)))
                elif self.mode == 'primitive':
                    category = image_id.split("_")[0]
                    plt.imshow(Image.open(os.path.join(configs.PRIMITIVE_ROOT, "{}/{}.png".format(category, image_id))).convert('RGB').resize((224, 224)))
                plt.text(240, 60, 'beam size 1 : ' + self.chosen['1'][q][i][1], fontsize=14)
                plt.text(240, 90, 'beam size 3 : ' + self.chosen['3'][q][i][1], fontsize=14)
                plt.text(240, 120, 'beam size 5 : ' + self.chosen['5'][q][i][1], fontsize=14)
                plt.text(240, 150, 'beam size 7 : ' + self.chosen['7'][q][i][1], fontsize=14)
                plt.text(240, 180, 'ground truth : ' + self.chosen['7'][q][i][2], fontsize=14)
                plt.axis('off')
            if not os.path.exists("outputs/figs/{}".format(path)):
                os.mkdir("outputs/figs/{}".format(path))
            plt.savefig("outputs/figs/{}/{}.png".format(path, q), bbox_inches="tight")


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
    num = args.num
    encoder_path = args.encoder
    decoder_path = args.decoder
    outname = encoder_path[8:-4]
    print("\n[settings]")
    print("GPU:", args.gpu)
    print("dataset:", args.dataset)
    print("train_size:", args.train_size)
    print("test_size:", args.test_size)
    print("batch_size:", args.batch_size)
    print("outname:", outname)
    print()
    print("preparing data...")
    print()
    if args.dataset == 'shapenet':
        embeddings = PretrainedEmbeddings(
            [
                pickle.load(open(configs.SHAPENET_EMBEDDING.format("train"), 'rb')),
                pickle.load(open(configs.SHAPENET_EMBEDDING.format("val"), 'rb')),
                pickle.load(open(configs.SHAPENET_EMBEDDING.format("test"), 'rb')),
            ],
            [
                train_size,
                0,
                test_size
            ],
            configs.MAX_LENGTH
        )
    elif args.dataset == 'primitive':
        embeddings = PretrainedEmbeddings(
            [
                pickle.load(open(configs.PRIMITIVE_EMBEDDING.format("train"), 'rb')),
                pickle.load(open(configs.PRIMITIVE_EMBEDDING.format("val"), 'rb')),
                pickle.load(open(configs.PRIMITIVE_EMBEDDING.format("test"), 'rb')),
            ],
            [
                train_size,
                0,
                test_size
            ],
            configs.MAX_LENGTH
        )
    else:
        print("invalid dataset, terminating...")
        return
    dict_idx2word = embeddings.dict_idx2word
    corpus = embeddings.test_ref
    test_ds = EmbeddingCaptionDataset(
        embeddings.test_shape_embeddings
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_ec, shuffle=True)
    print("initializing models...")
    print()
    encoder = torch.load(os.path.join("outputs/models", encoder_path)).cuda()
    decoder = torch.load(os.path.join("outputs/models", decoder_path)).cuda()
    encoder.eval()
    decoder.eval()
    beam_size = ['1', '3', '5', '7']
    candidates = {i:{} for i in beam_size}
    outputs = {i:{} for i in beam_size}
    bleu = {i:{} for i in beam_size}
    cider = {i:{} for i in beam_size}
    rouge = {i:{} for i in beam_size}
    if os.path.exists("outputs/results/{}.json".format(outname)):
        print("loading existing results...")
        print()
        candidates = json.load(open("outputs/results/{}.json".format(outname)))
    else:
        print("\nevaluating with beam search...")
        print()
        for _, (model_ids, _, embeddings, lengths) in enumerate(test_dl):
            visual_inputs = Variable(embeddings, requires_grad=False).cuda()
            cap_lengths = Variable(lengths, requires_grad=False).cuda()
            visual_contexts = encoder(visual_inputs)
            max_length = int(cap_lengths[0].item()) + 10
            for bs in beam_size:
                outputs[bs] = decoder.beam_search(visual_contexts, int(bs), max_length)
                outputs[bs] = decode_outputs(outputs[bs], None, dict_idx2word, "val")
                for model_id, output in zip(model_ids, outputs[bs]):
                    if model_id not in candidates[bs].keys():
                        candidates[bs][model_id] = [output]
                    else:
                        candidates[bs][model_id].append(output)
        # save results
        json.dump(candidates, open("outputs/results/{}.json".format(outname), 'w'))

    with open("outputs/scores/{}.txt".format(outname), 'w') as f:
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
            # write report
            f.write("----------------------Beam_size: {}-----------------------\n".format(bs))
            f.write("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[bs][0][0], max(bleu[bs][1][0]), min(bleu[bs][1][0])))
            f.write("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[bs][0][1], max(bleu[bs][1][1]), min(bleu[bs][1][1])))
            f.write("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[bs][0][2], max(bleu[bs][1][2]), min(bleu[bs][1][2])))
            f.write("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[bs][0][3], max(bleu[bs][1][3]), min(bleu[bs][1][3])))
            f.write("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(cider[bs][0], max(cider[bs][1]), min(cider[bs][1])))
            f.write("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n\n".format(rouge[bs][0], max(rouge[bs][1]), min(rouge[bs][1])))


   # save figs
    report = Report(corpus, candidates, cider, args.dataset, num)
    report(outname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='shapenet', help="shapenet/primitive")
    parser.add_argument("--train_size", type=int, default=100, help="train size for input captions")
    parser.add_argument("--test_size", type=int, default=10, help="test size for input captions")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--num", type=int, default=3, help="number of shown images")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    parser.add_argument("--encoder", type=str, default=None, help="path to encoder")
    parser.add_argument("--decoder", type=str, default=None, help="path to decoder")
    args = parser.parse_args()
    main(args)