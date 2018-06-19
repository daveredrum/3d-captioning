import configs
import pickle
import pandas
import numpy
import os
import json
import random
import capeval.bleu.bleu as capbleu
import capeval.cider.cider as capcider
from data import PretrainedEmbeddings

if __name__ == "__main__":
    embeddings = PretrainedEmbeddings(
        [
            pickle.load(open(configs.PROCESSED_SHAPE_EMBEDDING.format("train"), 'rb'))['caption_embedding_tuples'],
            pickle.load(open(configs.PROCESSED_SHAPE_EMBEDDING.format("val"), 'rb'))['caption_embedding_tuples'],
            pickle.load(open(configs.PROCESSED_SHAPE_EMBEDDING.format("test"), 'rb'))['caption_embedding_tuples'],
        ],
        [
            -1,
            0,
            -1
        ],
        json.load(open(os.path.join(configs.DATA_ROOT, "shapenet.json")))['idx_to_word'],
        18
    )
    train_embeddings = embeddings.train_embeddings
    test_embeddings = embeddings.test_embeddings
    train_ref = embeddings.train_ref
    test_ref = embeddings.test_ref
    # get candidates
    print("\ntesting...\n")
    test_can = {}
    for test_item in test_embeddings:
        best_sim = 0
        best_match = None
        for train_item in train_embeddings:
            sim = test_item[2].reshape((1, 128)).dot(train_item[2].reshape((128, 1)))[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_match = ' '.join([embeddings.dict_idx2word[str(index)] for index in train_item[1]])
        if test_item[0] in test_can.keys():
            test_can[test_item[0]].append(best_match)
        else:
            test_can[test_item[0]] = [best_match]
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