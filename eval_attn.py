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
from model.encoder_attn import *
from model.encoder_shape import ShapenetShapeEncoder
from model.encoder_text import ShapenetTextEncoder
from lib.configs import CONF

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

def _get_modelid2idx(common_model_ids):
    modelid2idx = {}
    for idx, modelid in enumerate(common_model_ids):
        modelid2idx[modelid] = idx
    
    return modelid2idx

def _filter_label(model_ids, common_model_ids, modelid2idx):
    filtered = []
    for model_id in model_ids:
        if model_id in common_model_ids:
            filtered.append(modelid2idx[model_id])
    
    return filtered

def _feed(shape_encoder, text_encoder, shapeloader, textloader, batch_size, verbose, mode, is_multihead):
    if mode == 't2s':
        a_dataloader, b_dataloader = textloader, shapeloader
    elif mode == 's2t':
        a_dataloader, b_dataloader = shapeloader, textloader
    else:
        raise ValueError("invalid mode, terminating...")

    a_model_id = []
    for model_id_a, _, _, _, _, _ in a_dataloader:
        for i in range(batch_size):
            a_model_id.append(model_id_a[i])

    b_model_id = []
    for model_id_b, _, _, _, _, _ in b_dataloader:
        for i in range(batch_size):
            b_model_id.append(model_id_b[i])

    common_model_ids = list(set(a_model_id).intersection(b_model_id))
    modelid2idx = _get_modelid2idx(common_model_ids)
    a_label = _filter_label(a_model_id, common_model_ids, modelid2idx)
    b_label = _filter_label(b_model_id, common_model_ids, modelid2idx)
    num_a = len(a_label)
    num_b = len(b_label)

    if verbose > 0:
        print("evaluate on {} models\n".format(len(common_model_ids)))

    sim_a2b = np.zeros((num_a, num_b))
    exe_s = []
    total_iter = len(a_dataloader)
    offset_a = 0
    for a_id, (model_id_a, a_shape, a_text, _, _, _) in enumerate(a_dataloader):
        sbt = time.time()
        offset_b = 0
        for b_id, (model_id_b, b_shape, b_text, _, _, _) in enumerate(b_dataloader):
            if text_encoder:
                if mode == 't2s':
                    a = a_text.cuda()
                    b = b_shape.cuda()
                    if isinstance(shape_encoder, SelfAttnShapeEncoder) and isinstance(text_encoder, SelfAttnTextEncoder):
                        a_embedding, _ = text_encoder(a)
                        b_embedding, _ = shape_encoder(b)
                    else:
                        a_embedding, b_embedding = text_encoder(a), shape_encoder(b)
                elif mode == 's2t':
                    a = a_shape.cuda()
                    b = b_text.cuda()
                    if isinstance(shape_encoder, SelfAttnShapeEncoder) and isinstance(text_encoder, SelfAttnTextEncoder):
                        a_embedding, _ = shape_encoder(a)
                        b_embedding, _ = text_encoder(b)
                    else:
                        a_embedding, b_embedding = shape_encoder(a), text_encoder(b)
                else:
                    raise ValueError("invalid mode, terminating...")
            else:
                if mode == 't2s':
                    a = a_text.cuda()
                    b = b_shape.cuda() 
                    if is_multihead:
                        b_embedding, a_embedding, _, _, _, _ = shape_encoder(b, a)
                    else:
                        b_embedding, a_embedding, _, _ = shape_encoder(b, a)
                elif mode == 's2t':
                    a = a_shape.cuda()
                    b = b_text.cuda()
                    if is_multihead:
                        a_embedding, b_embedding, _, _, _, _ = shape_encoder(a, b)
                    else:
                        a_embedding, b_embedding, _, _ = shape_encoder(a, b)
                else:
                    raise ValueError("invalid mode, terminating...")

            sim = a_embedding.matmul(b_embedding.transpose(1, 0))
            offset_in_batch_a = 0
            for i in range(batch_size):
                if model_id_a[i] in common_model_ids:
                    offset_in_batch_b = 0
                    for j in range(batch_size):
                        if model_id_b[j] in common_model_ids:
                            sim_a2b[offset_a + offset_in_batch_a, offset_b + offset_in_batch_b] = sim[i, j].item()
                            offset_in_batch_b += 1
                    
                    offset_in_batch_a += 1
            
            offset_b += sum([1 for model_id in model_id_b if model_id in common_model_ids])
        
        offset_a += sum([1 for model_id in model_id_a if model_id in common_model_ids])
        
        exe_s.append(time.time() - sbt)

        if verbose > 0 and (a_id + 1) % verbose == 0:
            avg_exe_s = np.mean(exe_s)
            eta_s = avg_exe_s * (total_iter - a_id - 1)
            eta_m = math.floor(eta_s / 60)
            print("complete step no.{}, {} left, ETA: {}m {}s".format(a_id + 1, len(a_dataloader) - a_id - 1, eta_m, int(eta_s - eta_m * 60)))

    return sim_a2b, a_label, b_label

def retrieve(shape_encoder, text_encoder, shapeloader, textloader, batch_size, verbose, is_multihead):
    # t2s
    if verbose != 0:
        print("retrieve text-to-shape\n")
    sim_t2s, source_t2s, target_t2s = _feed(shape_encoder, text_encoder, shapeloader, textloader, batch_size, verbose, 't2s', is_multihead)

    # s2t
    if verbose != 0:
        print("\nretrieve shape-to-text\n")
    sim_s2t, source_s2t, target_s2t = _feed(shape_encoder, text_encoder, shapeloader, textloader, batch_size, verbose, 's2t', is_multihead)

    return sim_t2s, sim_s2t, source_t2s, source_s2t, target_t2s, target_s2t

def _check_multihead(version):
    flag = False
    if version and version == '3':
        flag = True
    
    return flag

def evaluate(
    shape_encoder, 
    text_encoder,
    shapeloader, 
    textloader, 
    batch_size,
    verbose,
    is_multihead
    ):
    # retrieve
    sim_t2s, sim_s2t, label_t2s, label_s2t, target_t2s, target_s2t = retrieve(
        shape_encoder, 
        text_encoder,
        shapeloader, 
        textloader, 
        batch_size,
        verbose,
        is_multihead
    )

    # compute scores
    n_neighbors = CONF.TRAIN.N_NEIGHBOR
    print("\ncompute text-to-shape retrieval scores...\n")
    indices_t2s = compute_nearest_neighbors_cosine(sim_t2s, n_neighbors)
    metrics_t2s = compute_pr_at_k(indices_t2s, label_t2s, n_neighbors, len(label_t2s), fit_labels=target_t2s)

    print("compute shape-to-text retrieval scores...\n")
    indices_s2t = compute_nearest_neighbors_cosine(sim_s2t, n_neighbors)
    metrics_s2t = compute_pr_at_k(indices_s2t, label_s2t, n_neighbors, len(label_s2t), fit_labels=target_s2t)

    return metrics_t2s, metrics_s2t

def main(args):
    # parse args
    root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path)
    voxel = int(args.path.split("_")[1][1:])
    
    phase = args.phase
    batch_size = args.batch_size
    size = args.size
    gpu = args.gpu
    verbose = args.verbose
    attention = args.path.split("_")[-1]
    if attention != 'noattention':
        version = attention[8:]
    else:
        version = None
    is_multihead = _check_multihead(version)

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
        batch_size,
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
    textloader = DataLoader(textset, batch_size=batch_size, collate_fn=collate_shapenet, drop_last=True)
    shapeloader = DataLoader(shapeset, batch_size=batch_size, collate_fn=collate_shapenet, drop_last=True)
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
    if is_multihead:
        print("\ninitializing multi-head models...\n")
        shape_encoder_path = os.path.join(root, "models/encoder.pth")
        shape_encoder = MultiHeadEncoder(shapenet.dict_idx2word.__len__()).cuda()
        shape_encoder.load_state_dict(torch.load(shape_encoder_path))
        shape_encoder.eval()
        text_encoder = None
    else:
        if attention == 'noattention':
            print("\ninitializing naive models...\n")
            shape_encoder_path = os.path.join(root, "models/shape_encoder.pth")
            text_encoder_path = os.path.join(root, "models/text_encoder.pth")
            shape_encoder = ShapenetShapeEncoder().cuda()
            text_encoder = ShapenetTextEncoder(shapenet.dict_idx2word.__len__()).cuda()
            shape_encoder.load_state_dict(torch.load(shape_encoder_path))
            text_encoder.load_state_dict(torch.load(text_encoder_path))
            shape_encoder.eval()
            text_encoder.eval()
        elif attention == 'self':
            print("\ninitializing self-attentive models...\n")
            shape_encoder_path = os.path.join(root, "models/shape_encoder.pth")
            text_encoder_path = os.path.join(root, "models/text_encoder.pth")
            shape_encoder = SelfAttnShapeEncoder().cuda()
            text_encoder = SelfAttnTextEncoder(shapenet.dict_idx2word.__len__()).cuda()
            shape_encoder.load_state_dict(torch.load(shape_encoder_path))
            text_encoder.load_state_dict(torch.load(text_encoder_path))
            shape_encoder.eval()
            text_encoder.eval()
        else:
            print("\ninitializing attentive models ver.{}...\n".format(version))
            shape_encoder_path = os.path.join(root, "models/encoder.pth")
            shape_encoder = AdaptiveEncoder(shapenet.dict_idx2word.__len__(), version).cuda()
            shape_encoder.load_state_dict(torch.load(shape_encoder_path))
            shape_encoder.eval()
            text_encoder = None

    evaluate(shape_encoder, text_encoder, shapeloader, textloader, batch_size, verbose, is_multihead)


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
