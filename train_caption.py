import pandas 
import numpy as np
import os
import re
import operator
import math
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from lib.configs import CONF
import lib.capeval.bleu.bleu as capbleu
import lib.capeval.cider.cider as capcider
import lib.capeval.meteor.meteor as capmeteor
import lib.capeval.rouge.rouge as caprouge
from lib.data_caption import *
from model.encoders_caption import *
from model.decoders_caption import *
from lib.solver_caption import *
import matplotlib.pyplot as plt
from lib.utils import draw_curves_caption
from lib.ext_embedding import parse_path

def get_dataset(path):
    embeddings = PretrainedEmbeddings(pickle.load(open(path, 'rb')))
    dataset = {
        'train': CaptionDataset(embeddings.train_text, embeddings.train_shape),
        'val': CaptionDataset(embeddings.val_text, embeddings.val_shape)
    }

    return dataset, embeddings

def get_dataloader(dataset, batch_size):
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_ec),
        'val': DataLoader(dataset['val'], batch_size=batch_size, shuffle=True, collate_fn=collate_ec)
    }
    
    return dataloader

def get_reference(embeddings):
    references = {
        'train': embeddings.train_ref,
        'val': embeddings.val_ref
    }
    # load vocabulary
    dict_idx2word = embeddings.dict_idx2word
    dict_word2idx = embeddings.dict_word2idx
    dict_size = len(dict_idx2word)

    return references, dict_idx2word, dict_word2idx, dict_size

def get_models(attention, batch_size, dict_size, embeddings):
    if attention == "fc":
        print('initializing models without attention...\n')
        encoder = EmbeddingEncoder().cuda()
        decoder = Decoder(dict_size, CONF.CAP.HIDDEN_SIZE).cuda()
    elif attention:
        print('initializing models with attention {}...\n'.format(attention))
        encoder = AttentionEncoder(embeddings.visual_channel).cuda()
        decoder = AttentionDecoder3D(
            attention, 
            batch_size, 
            dict_size, 
            CONF.CAP.HIDDEN_SIZE, 
            embeddings.visual_channel, 
            embeddings.visual_size
        ).cuda()

    return encoder, decoder

def get_optimizer(encoder, decoder, learning_rate, weight_decay):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return criterion, optimizer

def get_settings(train_size, val_size, epoch, dict_size, embedding_settings):
    settings = "{}_{}_{}_trs{}_vs{}_e{}_lr{:0.5f}_w{:0.5f}_bs{}_vocab{}_beam{}".format(
        CONF.CAP.DATASET,
        embedding_settings['attention_type'],
        CONF.CAP.ATTN, 
        train_size, 
        val_size, 
        epoch, 
        CONF.CAP.LEARNING_RATE, 
        CONF.CAP.WEIGHT_DECAY, 
        CONF.CAP.BATCH_SIZE, 
        dict_size, 
        CONF.CAP.BEAM_SIZE
    )

    return settings

def evaluate(encoder, decoder, dataloader, dict_idx2word, references, output_root):
    encoder.eval()
    decoder.eval()
    beam_size = ['1', '3', '5', '7']
    candidates = {i:{} for i in beam_size}
    outputs = {i:{} for i in beam_size}
    bleu = {i:{} for i in beam_size}
    cider = {i:{} for i in beam_size}
    rouge = {i:{} for i in beam_size}
    result_path = os.path.join(output_root, "results.json")
    if os.path.exists(result_path):
        print("loading existing results...")
        print()
        candidates = json.load(open(result_path, 'r'))
    else:
        print("evaluating with beam search...")
        print()
        for _, (model_ids, captions, embeddings, embeddings_interm, lengths) in enumerate(dataloader[CONF.CAP.EVAL_DATASET]):
            if CONF.CAP.ATTN == 'fc':
                visual_inputs = embeddings.cuda()
            else:
                visual_inputs = embeddings_interm.cuda()
            caption_inputs = captions[:, :-1].cuda()
            cap_lengths = lengths.cuda()
            visual_contexts = encoder(visual_inputs)
            max_length = int(cap_lengths[0].item()) + 10
            for bs in beam_size:
                if CONF.CAP.ATTN == 'fc':
                    outputs[bs] = decoder.beam_search(visual_contexts, int(bs), max_length)
                    outputs[bs] = decode_outputs(outputs[bs], None, dict_idx2word, "val")
                else:
                    outputs[bs] = decoder.beam_search(visual_contexts, caption_inputs, int(bs), max_length)
                    outputs[bs] = decode_attention_outputs(outputs[bs], None, dict_idx2word, "val")
                for model_id, output in zip(model_ids, outputs[bs]):
                    if model_id not in candidates[bs].keys():
                        candidates[bs][model_id] = [output]
                    else:
                        candidates[bs][model_id].append(output)
        # save results
        json.dump(candidates, open(result_path, 'w'))

    score_path = os.path.join(output_root, "scores.txt")
    with open(score_path, 'w') as f:
        for bs in beam_size:
            # compute
            bleu[bs] = capbleu.Bleu(4).compute_score(references[CONF.CAP.EVAL_DATASET], candidates[bs])
            cider[bs] = capcider.Cider().compute_score(references[CONF.CAP.EVAL_DATASET], candidates[bs])
            rouge[bs] = caprouge.Rouge().compute_score(references[CONF.CAP.EVAL_DATASET], candidates[bs])
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

def main(args):
    # settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    epoch = args.epoch
    beam_size = CONF.CAP.BEAM_SIZE
    verbose = CONF.CAP.VERBOSE
    learning_rate = CONF.CAP.LEARNING_RATE
    batch_size = CONF.CAP.BATCH_SIZE
    weight_decay = CONF.CAP.WEIGHT_DECAY
    attention = CONF.CAP.ATTN
    embedding_path = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path, "embedding/embedding.p")
    embedding_settings = parse_path(args.path)
    
    # data
    print('\npreparing data...\n')
    datasets, embeddings = get_dataset(embedding_path)
    dataloaders = get_dataloader(datasets, batch_size)
    references, dict_idx2word, dict_word2idx, dict_size = get_reference(embeddings)
    encoder, decoder = get_models(attention, batch_size, dict_size, embeddings)
    criterion, optimizer = get_optimizer(encoder, decoder, learning_rate, weight_decay)

    # train
    print("[settings]")
    print("GPU:", args.gpu)
    print("dataset:", CONF.CAP.DATASET)
    print("train_size:", embeddings.train_size)
    print("val_size:", embeddings.val_size)
    print("beam_size:", CONF.CAP.BEAM_SIZE)
    print("epoch:", args.epoch)
    print("verbose:", CONF.CAP.VERBOSE)
    print("batch_size:", CONF.CAP.BATCH_SIZE)
    print("learning_rate:", learning_rate)
    print("weight_decay:", weight_decay)
    print("dict_size:", dict_size)
    print("attention:", attention)
    print("embedding:", embedding_settings['attention_type'])
    print("evaluation:", CONF.CAP.IS_EVAL)
    print()
    
    settings = get_settings(embeddings.train_size, embeddings.val_size, epoch, dict_size, embedding_settings)
    output_root = os.path.join(CONF.PATH.OUTPUT_CAPTION, settings)
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    # encoder_decoder_solver = EncoderDecoderSolver(
    #     optimizer,
    #     criterion,
    #     output_root
    # )
    # encoder, decoder = encoder_decoder_solver.train(
    #     encoder,
    #     decoder,
    #     dataloaders,
    #     references,
    #     dict_word2idx,
    #     dict_idx2word,
    #     epoch,
    #     verbose,
    #     attention,
    #     beam_size
    # )

    # draw_curves_caption(encoder_decoder_solver, output_root)
    # pickle.dump(encoder_decoder_solver.log, open(os.path.join(output_root, "log.p"), 'wb'))

    if CONF.CAP.IS_EVAL:
        evaluate(encoder, decoder, dataloaders, dict_idx2word, references, output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the folder")
    parser.add_argument("--epoch", type=int, default=50, help="epochs for training")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)
    