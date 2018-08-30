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
from lib.configs import *
from lib.eval_embedding import compute_pr_at_k
from model.encoder_attn import AdaptiveEncoder

def compute_nearest_neighbors_cosine(similarities, n_neighbors, range_start=0):
    '''
        derived from https://github.com/kchen92/text2shape/blob/master/tools/eval/eval_text_encoder.py
    '''
    
    n_neighbors += 1

    # Argpartition method
    unnormalized_similarities = similarities
    n_samples = unnormalized_similarities.shape[0]
    sort_indices = np.argpartition(unnormalized_similarities, -n_neighbors, axis=1)
    indices = sort_indices[:, -n_neighbors:]
    row_indices = [x for x in range(n_samples) for _ in range(n_neighbors)]
    yo = unnormalized_similarities[row_indices, indices.flatten()].reshape(n_samples, n_neighbors)
    indices = indices[row_indices, np.argsort(yo, axis=1).flatten()].reshape(n_samples, n_neighbors)
    indices = np.flip(indices, 1)

    n_neighbors -= 1  # Undo the neighbor increment

    final_indices = np.zeros((indices.shape[0], n_neighbors), dtype=int)
    compare_mat = np.asarray(list(range(range_start, range_start + indices.shape[0]))).reshape(indices.shape[0], 1)
    has_self = np.equal(compare_mat, indices)  # has self as nearest neighbor
    any_result = np.any(has_self, axis=1)
    for row_idx in range(indices.shape[0]):
        if any_result[row_idx]:
            nonzero_idx = np.nonzero(has_self[row_idx, :])
            assert len(nonzero_idx) == 1
            new_row = np.delete(indices[row_idx, :], nonzero_idx[0])
            final_indices[row_idx, :] = new_row
        else:
            final_indices[row_idx, :] = indices[row_idx, :n_neighbors]
    indices = final_indices
    
    return indices

def _align_label(idx2label):
    new_idx2label = {}
    for idx, (key, _) in enumerate(idx2label.items()):
        new_idx2label[key] = idx
    
    return new_idx2label

def _filter_label(labels, common_model_ids):
    filtered = []
    for l in labels:
        if l in common_model_ids:
            filtered.append(l)
    
    return filtered

def _feed(encoder, src_dataloader, tar_dataloader, num_src, num_tar, idx2label, batch_size, verbose):
    assert len(src_dataloader) * batch_size == num_src
    assert len(tar_dataloader) * batch_size == num_tar

    src_label = []
    for model_id_src, _, _, _, _, _ in src_dataloader:
        for i in range(batch_size):
            src_label.append(idx2label[model_id_src[i]])

    tar_label = []
    for model_id_tar, _, _, _, _, _ in tar_dataloader:
        for i in range(batch_size):
            tar_label.append(idx2label[model_id_tar[i]])

    num_matched = min(len(set(src_label)), len(set(tar_label)))
    common_model_ids = list(set(src_label).intersection(tar_label))
    src_label = _filter_label(src_label, common_model_ids)
    tar_label = _filter_label(tar_label, common_model_ids)
    num_src = len(src_label)
    num_tar = len(tar_label)
    assert len(set(src_label).intersection(tar_label)) == num_matched

    sim_src2tar = np.zeros((num_src, num_tar))
    exe_s = []
    total_iter = len(src_dataloader)
    for src_id, (model_id_src, _, src, _, _, _) in enumerate(src_dataloader):
        start = time.time()
        for tar_id, (model_id_tar, tar, _, _, _, _) in enumerate(tar_dataloader):
            src = src.cuda()
            tar = tar.cuda()
            tar_embedding, src_embedding , _, _ = encoder(tar, src)
            sim = src_embedding.matmul(tar_embedding.transpose(1, 0))
            offset_i = 0
            for i in range(batch_size):
                offset_j = 0
                if idx2label[model_id_src[i]] in common_model_ids:
                    for j in range(batch_size):
                        if idx2label[model_id_tar[j]] in common_model_ids:
                            sim_src2tar[offset_i, offset_j] = sim[i, j].item()
                            offset_j += 1
                    offset_i += 1
        
        exe_s.append(time.time() - start)

        if (src_id + 1) % verbose == 0:
            avg_exe_s = np.mean(exe_s)
            eta_s = avg_exe_s * (total_iter - src_id - 1)
            eta_m = math.floor(eta_s / 60)
            print("complete step no.{}, {} left, ETA: {}m {}s".format(src_id + 1, len(src_dataloader) - src_id - 1, eta_m, int(eta_s - eta_m * 60)))

    return sim_src2tar, src_label, tar_label

def retrieve(encoder, shapeloader, textloader, num_shape, num_text, idx2label, batch_size, verbose):
    # t2s
    print("retrieve text-to-shape\n")
    sim_t2s, source_t2s, target_t2s = _feed(encoder, textloader, shapeloader, num_text, num_shape, idx2label, batch_size, verbose)

    # s2t
    print("\nretrieve shape-to-text\n")
    sim_s2t, source_s2t, target_s2t = _feed(encoder, shapeloader, textloader, num_shape, num_text, idx2label, batch_size, verbose)

    return sim_t2s, sim_s2t, source_t2s, source_s2t, target_t2s, target_s2t


def main(args):
    # parse args
    root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path)
    voxel = int(args.path.split("_")[1][1:])
    encoder_path = os.path.join(root, "models/encoder.pth")
    
    phase = args.phase
    batch_size = args.batch_size
    size = args.size
    gpu = args.gpu
    verbose = args.verbose

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
    textset = ShapenetDataset(
        getattr(shapenet, "{}_data".format(phase)), 
        getattr(shapenet, "{}_idx2label".format(phase)), 
        getattr(shapenet, "{}_label2idx".format(phase)), 
        voxel,
        h5py.File(CONF.PATH.SHAPENET_DATABASE.format(voxel), "r")
    )
    shapeset = ShapenetDataset(
        getattr(shapenet, "{}_data".format(phase)), 
        getattr(shapenet, "{}_idx2label".format(phase)), 
        getattr(shapenet, "{}_label2idx".format(phase)), 
        voxel,
        h5py.File(CONF.PATH.SHAPENET_DATABASE.format(voxel), "r"),
        aggr_shape=True
    )
    textloader = DataLoader(textset, batch_size=batch_size, shuffle=False, collate_fn=collate_shapenet, drop_last=True)
    shapeloader = DataLoader(shapeset, batch_size=batch_size, shuffle=False, collate_fn=collate_shapenet, drop_last=True)
    num_shape = len(shapeloader) * batch_size
    num_text = len(textloader) * batch_size

    # report settings
    print("[settings]")
    print("evaluate on {} set".format(phase))
    print("shape embeddings: {} -> eval on {} samples".format(len(shapeset), num_shape))
    print("text embeddings: {} -> eval on {} samples".format(len(textset), num_text))
    print("batch size:", batch_size)
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    encoder = AdaptiveEncoder(shapenet.dict_idx2word.__len__(), args.path.split("_")[-1][8:]).cuda()
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    # retrieve
    idx2label = _align_label(getattr(shapenet, "{}_idx2label".format(phase)))
    sim_t2s, sim_s2t, label_t2s, label_s2t, target_t2s, target_s2t = retrieve(
        encoder, 
        shapeloader, 
        textloader, 
        num_shape, 
        num_text, 
        idx2label, 
        batch_size,
        verbose
    )

    # compute scores
    n_neighbors = 20
    print("\ncompute text-to-shape retrieval scores...\n")
    indices_t2s = compute_nearest_neighbors_cosine(sim_t2s, n_neighbors)
    compute_pr_at_k(indices_t2s, label_t2s, n_neighbors, len(label_t2s), fit_labels=target_t2s)

    print("compute shape-to-text retrieval scores...\n")
    indices_s2t = compute_nearest_neighbors_cosine(sim_s2t, n_neighbors)
    compute_pr_at_k(indices_s2t, label_s2t, n_neighbors, len(label_s2t), fit_labels=target_s2t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--phase", type=str, default='val', help="train/val/test")
    parser.add_argument("--size", type=int, default=-1, help="size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    args = parser.parse_args()
    main(args)
