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
import configs
from data import *
from encoders import *
from decoders import *
from solver import *
import matplotlib.pyplot as plt


def main(args):
    # settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    beam_size = args.beam_size
    epoch = args.epoch
    verbose = args.verbose
    lr = args.learning_rate
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    if args.evaluation == "true":
        evaluation = True
    elif args.evaluation == "false":
        evaluation = False

    ###################################################################
    #                                                                 #
    #                                                                 #
    #                        training for decoder                     #
    #                                                                 #
    #                                                                 #
    ###################################################################
    
    # embeddings
    if args.dataset == 'shapenet':
        embeddings = PretrainedEmbeddings(
            [
                pickle.load(open(configs.SHAPENET_EMBEDDING.format("train"), 'rb')),
                pickle.load(open(configs.SHAPENET_EMBEDDING.format("val"), 'rb')),
                pickle.load(open(configs.SHAPENET_EMBEDDING.format("test"), 'rb')),
            ],
            [
                train_size,
                val_size,
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
                val_size,
                test_size
            ],
            configs.MAX_LENGTH
        )
    else:
        print("invalid dataset, terminating...")
        return

    # data settings
    train_ds = EmbeddingCaptionDataset(
        embeddings.train_embeddings
    )
    val_ds = EmbeddingCaptionDataset(
        embeddings.val_shape_embeddings
    )
    test_ds = EmbeddingCaptionDataset(
        embeddings.test_shape_embeddings
    )
    dataloader = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ec),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ec),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ec)
    }
    references = {
        'train': embeddings.train_ref,
        'val': embeddings.val_ref,
        'test': embeddings.test_ref
    }
    # load vocabulary
    dict_idx2word = embeddings.dict_idx2word
    dict_word2idx = embeddings.dict_word2idx
    input_size = len(dict_idx2word)

    # initialize the models
    encoder = EmbeddingEncoder().cuda()
    decoder = Decoder(
        input_size,
        512
    ).cuda()

    # initialize the optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # train
    print("\n[settings]")
    print("GPU:", args.gpu)
    print("dataset:", args.dataset)
    print("train_size:", args.train_size)
    print("val_size:", args.val_size)
    print("test_size:", args.test_size)
    print("beam_size:", args.beam_size)
    print("epoch:", args.epoch)
    print("verbose:", args.verbose)
    print("batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("weight_decay:", args.weight_decay)
    print("vocabulary:", input_size)
    print("evaluation:", args.evaluation)
    print()
    settings = "{}_trs{}_vs{}_ts{}_e{}_lr{:0.5f}_w{:0.5f}_bs{}_vocab{}_beam{}".format(args.dataset, train_size, val_size, test_size, epoch, lr, weight_decay, batch_size, input_size, beam_size)
    encoder_decoder_solver = EncoderDecoderSolver(
        optimizer,
        criterion,
        settings
    )
    encoder, decoder = encoder_decoder_solver.train(
        encoder,
        decoder,
        dataloader,
        references,
        dict_word2idx,
        dict_idx2word,
        epoch,
        verbose,
        beam_size
    )

    ###################################################################
    #                                                                 #
    #                                                                 #
    #                             evaluation                          #
    #                                                                 #
    #                                                                 #
    ###################################################################
    if evaluation:
        encoder.eval()
        decoder.eval()
        beam_size = ['1', '3', '5', '7']
        candidates = {i:{} for i in beam_size}
        outputs = {i:{} for i in beam_size}
        bleu = {i:{} for i in beam_size}
        cider = {i:{} for i in beam_size}
        rouge = {i:{} for i in beam_size}
        if os.path.exists("scores/{}.json".format(settings)):
            print("loading existing results...")
            print()
            candidates = json.load(open("scores/{}.json".format(settings)))
        else:
            print("evaluating with beam search...")
            print()
            for _, (model_ids, _, embeddings, lengths) in enumerate(dataloader['test']):
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
            json.dump(candidates, open("scores/{}.json".format(settings), 'w'))

        for bs in beam_size:
            # compute
            bleu[bs] = capbleu.Bleu(4).compute_score(references['test'], candidates[bs])
            cider[bs] = capcider.Cider().compute_score(references['test'], candidates[bs])
            rouge[bs] = caprouge.Rouge().compute_score(references['test'], candidates[bs])
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
    parser.add_argument("--dataset", type=str, default='shapenet', help="shapenet/primitive")
    parser.add_argument("--train_size", type=int, default=100, help="train size for input captions")
    parser.add_argument("--val_size", type=int, default=10, help="val size for input captions")
    parser.add_argument("--test_size", type=int, default=10, help="test size for input captions")
    parser.add_argument("--beam_size", type=int, default=1, help="beam size")
    parser.add_argument("--epoch", type=int, default=100, help="epochs for training")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="penalty on the optimizer")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    parser.add_argument("--evaluation", type=str, default="false", help="true/false")
    args = parser.parse_args()
    main(args)
    