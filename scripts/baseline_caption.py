import pickle
import pandas
import numpy as np
import os
import json
import time
import random
import argparse
from queue import Queue
from numpy.linalg import norm
import multiprocessing as mp

# HACK
import sys
sys.path.append(".")
from lib.configs import CONF
import lib.capeval.bleu.bleu as capbleu
import lib.capeval.cider.cider as capcider


def get_embedding(path, phase):
    print("\ngetting embedding...") 
    embedding_path = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, path, "embedding/embedding.p")
    embedding = pickle.load(open(embedding_path, "rb"))[phase]
    keys = embedding.keys()
    key2label = {key: i for i, key in enumerate(list(keys))}
    label2key = {i: key for i, key in enumerate(list(keys))}

    return embedding, keys, key2label, label2key

def decode_embedding(embedding ,keys, key2label):
    print("decoding embedding...")
    shape_embeddings = [embedding[key]['shape_embedding'][0].reshape(1, -1) for key in keys]
    shape_labels = [key2label[key] for key in keys]
    text_embeddings = [item[1].reshape(1, -1) for key in keys for item in embedding[key]['text_embedding']]
    text_labels = [key2label[key] for key in keys for _ in embedding[key]['text_embedding']]
    text_raw = [item[0] for key in keys for item in embedding[key]['text_embedding']]

    shape_embeddings = np.concatenate(shape_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)


    return shape_embeddings, shape_labels, text_embeddings, text_labels, text_raw

def get_reference(embedding):
    print("building reference...")
    reference = {}
    for key in embedding.keys():
        for item in embedding[key]['text_embedding']:
            ref = item[0]
            ref = '<start> ' + ref + ' <end>'
            if key in reference.keys():
                reference[key].append(ref)
            else:
                reference[key] = [ref]


    return reference

def get_candidate(sim, label2key, text_raw):
    print("building candidate...")
    indices = np.argmax(sim, axis=1)
    candidate = {label2key[i]: [text_raw[indices[i]]] for i in range(sim.shape[0])} 

    return candidate

def main(args):
    embedding, keys, key2label, label2key = get_embedding(args.path, args.phase)
    shape_embeddings, shape_labels, text_embeddings, text_labels, text_raw = decode_embedding(embedding ,keys, key2label)
    reference = get_reference(embedding)
    sim = shape_embeddings.dot(text_embeddings.T)
    candidate = get_candidate(sim, label2key, text_raw)
    
    # compute metrics
    print("computing metrics\n")
    bleu, _ = capbleu.Bleu(4).compute_score(reference, candidate)
    cider, _ = capcider.Cider().compute_score(reference, candidate)
    # report
    print("[BLEU-1]: {}".format(bleu[0]))
    print("[BLEU-2]: {}".format(bleu[1]))
    print("[BLEU-3]: {}".format(bleu[2]))
    print("[BLEU-4]: {}".format(bleu[3]))
    print("[CIDEr]: {}".format(cider))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to the root folder containing embeddings')
    parser.add_argument('--phase', help='train/val/test', default='val', type=str)
    args = parser.parse_args()
    main(args)