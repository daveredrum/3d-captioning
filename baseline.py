import configs
import pickle
import pandas
import numpy
import os
import json
import time
import random
import argparse
from queue import Queue
import multiprocessing as mp
import capeval.bleu.bleu as capbleu
import capeval.cider.cider as capcider
from data import PretrainedEmbeddings
from numpy.linalg import norm

def match(embeddings, test_item, best):
    for train_item in embeddings.train_embeddings:
        sim = test_item[2].reshape((1, 128)).dot(train_item[2].reshape((128, 1)))[0, 0]
        sim /= (norm(test_item[2]) * norm(train_item[2]))
        if sim > best['sim']:
            best['sim'] = sim
            best['match'] = ' '.join([embeddings.dict_idx2word[str(index)] for index in train_item[1]])

def chunk(target, size):
    for i in range(0, len(target), size):
        yield target[i: i + size]

def main(args):
    # embeddings
    print("\npreparing data...\n")
    if args.dataset == 'shapenet':
        embeddings = PretrainedEmbeddings(
            [
                pickle.load(open(configs.SHAPENET_SHAPE_EMBEDDING.format("train"), 'rb')),
                pickle.load(open(configs.SHAPENET_SHAPE_EMBEDDING.format("val"), 'rb')),
                pickle.load(open(configs.SHAPENET_SHAPE_EMBEDDING.format("test"), 'rb')),
            ],
            [
                -1,
                0,
                -1
            ],
            configs.MAX_LENGTH
        )
    elif args.dataset == 'primitive':
        embeddings = PretrainedEmbeddings(
            [
                pickle.load(open(configs.PRIMITIVE_SHAPE_EMBEDDING.format("train"), 'rb')),
                pickle.load(open(configs.PRIMITIVE_SHAPE_EMBEDDING.format("val"), 'rb')),
                pickle.load(open(configs.PRIMITIVE_SHAPE_EMBEDDING.format("test"), 'rb')),
            ],
            [
                -1,
                0,
                -1
            ],
            configs.MAX_LENGTH
        )
    else:
        print("invalid dataset, terminating...")
        return
    test_embeddings = list(chunk(embeddings.test_embeddings, args.batch_size))
    test_ref = embeddings.test_ref
    test_can = {}
    # queue = mp.SimpleQueue()
    total_iter = len(test_embeddings)
    for test_id, test_embedding in enumerate(test_embeddings):
        print("[Info] step: {}/{}".format(test_id + 1, total_iter))
        start = time.time()
        for test_item in test_embedding:
            best = mp.Manager().dict()
            best['sim'] = 0
            best['match'] = None
            processes = [mp.Process(target=match, args=(embeddings, test_item, best)) for i in range(args.worker)]
            # print("[Info] starting processes...")
            for p in processes:
                p.start()

            # print("[Info] joining processes...")
            for p in processes:
                p.join()
            
            # print("[Info] recording...")
            if test_item[0] in test_can.keys():
                test_can[test_item[0]].append(best['match'])
            else:
                test_can[test_item[0]] = [best['match']]

        exe_s = time.time() - start
        print("[Info] time_per_step: {}s".format(int(exe_s)))
        eta = (total_iter - test_id) * exe_s
        if eta < 60:
            print("[Info] ETA: {}s\n".format(int(eta)))
        elif 60 <= eta < 60 * 60:
            print("[Info] ETA: {}m {}s\n".format(int(eta // 60), int(eta % 60)))
        else:
            print("[Info] ETA: {}h {}m {}s\n".format(int(eta // 3600), int(eta % 3600 // 60), int(eta % 3600 % 60)))

    # compute metrics
    print("computing metrics\n")
    bleu, _ = capbleu.Bleu(4).compute_score(test_ref, test_can)
    cider, _ = capcider.Cider().compute_score(test_ref, test_can)
    # report
    print("[BLEU-1]: {}".format(bleu[0]))
    print("[BLEU-2]: {}".format(bleu[1]))
    print("[BLEU-3]: {}".format(bleu[2]))
    print("[BLEU-4]: {}".format(bleu[3]))
    print("[CIDEr]: {}".format(cider))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='shapenet', help="shapenet/primitive")
    parser.add_argument("--worker", type=int, default=4, help="number of workers for multithreading")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    args = parser.parse_args()
    main(args)