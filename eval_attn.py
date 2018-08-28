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


def retrieve(encoder, dataloader, num_shape, num_text, idx2label):
    # t2s
    print("retrieve text-to-shape")
    sim_t2s = np.zeros((num_text, num_shape))
    label_t2s = []
    target_t2s = []
    for text_id, (model_id_text, _, text, _, _, _) in enumerate(dataloader):
        reached_id = []
        shape_id = 0
        for model_id_shape, shape, _, _, _, _ in dataloader:
            if model_id_shape[0] not in reached_id:
                reached_id.append(model_id_shape[0])
                if model_id_shape[0] not in target_t2s:
                    target_t2s.append(idx2label[model_id_shape[0]])
                text = text.cuda()
                shape = shape.cuda()
                shape_embedding, text_embedding , _, _ = encoder(shape, text)
                sim = text_embedding.matmul(shape_embedding.transpose(1, 0))
                sim_t2s[text_id, shape_id] = sim.item()
                shape_id += 1
        
        label_t2s.append(idx2label[model_id_text[0]])
    
    assert sim_t2s.shape[0] == num_text
    assert sim_t2s.shape[1] == num_shape

    # s2t
    print("retrieve shape-to-text\n")
    sim_s2t = np.zeros((num_shape, num_text))
    shape_id = 0
    label_s2t = []
    target_s2t = []
    reached_id = []
    for model_id_shape, shape, _, _, _, _ in dataloader:
        if model_id_shape[0] not in reached_id:
            reached_id.append(model_id_shape[0])
            for text_id, (model_id_text, _, text, _, _, _) in enumerate(dataloader):
                if model_id_text[0] not in target_t2s:
                    target_s2t.append(idx2label[model_id_text[0]])
                text = text.cuda()
                shape = shape.cuda()
                shape_embedding, text_embedding , _, _ = encoder(shape, text)
                sim = shape_embedding.matmul(text_embedding.transpose(1, 0))
                sim_s2t[shape_id, text_id] = sim.item()
            
            shape_id += 1
            label_s2t.append(idx2label[model_id_shape[0]])

    assert sim_s2t.shape[0] == num_shape 
    assert sim_s2t.shape[1] == num_text

    return sim_t2s, sim_s2t, label_t2s, label_s2t, target_t2s, target_s2t


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
    num_shape = len(getattr(shapenet, "{}_data_group".format(phase)).keys())
    num_text = len(getattr(shapenet, "{}_data".format(phase)))

    # report settings
    print("[settings]")
    print("evaluate on {} set".format(phase))
    print("shape embeddings:", num_shape)
    print("text embeddings:", num_text)
    print("size:", len(getattr(shapenet, "{}_data".format(phase))))
    print("gpu:", gpu)

    # initialize models
    print("\ninitializing models...\n")
    encoder = AdaptiveEncoder(shapenet.dict_idx2word.__len__(), args.path.split("_")[-1][8:]).cuda()
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    # retrieve
    sim_t2s, sim_s2t, label_t2s, label_s2t, target_t2s, target_s2t = retrieve(
        encoder, dataloader, num_shape, num_text, getattr(shapenet, "{}_idx2label".format(phase)))

    # compute scores
    n_neighbors = 20
    print("compute text-to-shape retrieval scores")
    indices_t2s = compute_nearest_neighbors_cosine(sim_t2s, n_neighbors)
    compute_pr_at_k(indices_t2s, label_t2s, n_neighbors, num_text, fit_labels=target_t2s)

    print("compute shape-to-text retrieval scores")
    indices_s2t = compute_nearest_neighbors_cosine(sim_s2t, n_neighbors)
    compute_pr_at_k(indices_s2t, label_s2t, n_neighbors, num_text, fit_labels=target_s2t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    parser.add_argument("--phase", type=str, default='val', help="train/val/test")
    parser.add_argument("--size", type=int, default=-1, help="size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)