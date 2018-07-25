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
import lib.configs as configs
import lib.capeval.bleu.bleu as capbleu
import lib.capeval.cider.cider as capcider
import lib.capeval.meteor.meteor as capmeteor
import lib.capeval.rouge.rouge as caprouge
from lib.data_caption import *
from model.encoders_caption import *
from model.decoders_caption import *
from lib.solver_caption import *
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
    attention = args.attention
    if args.attention == 'none':
        attention = None
        attention_s = 'noattention'
    else:
        attention = args.attention
        attention_s = args.attention
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
    print('\npreparing data...\n')
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
    if attention:
        print('initializing models with attention {}...\n'.format(attention))
        encoder = AttentionEncoder().cuda()
        decoder = AttentionDecoder3D(attention, batch_size, input_size, 512, 256, 4).cuda()
    else:
        print('initializing models without attention...\n')
        encoder = EmbeddingEncoder().cuda()
        decoder = Decoder(input_size, 512).cuda()

    # initialize the optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if attention:
        optimizer = optim.Adam(
            [
                {'params': list(encoder.parameters()), 'lr': 0.1 * lr},
                {'params': list(decoder.parameters())}
            ],
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

    # train
    print("[settings]")
    print("GPU:", args.gpu)
    print("dataset:", args.dataset)
    print("train_size:", len(train_ds))
    print("val_size:", len(val_ds))
    print("test_size:", len(test_ds))
    print("beam_size:", args.beam_size)
    print("epoch:", args.epoch)
    print("verbose:", args.verbose)
    print("batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("weight_decay:", args.weight_decay)
    print("vocabulary:", input_size)
    print("attention:", attention)
    print("evaluation:", evaluation)
    print()
    
    settings = "{}_{}_trs{}_vs{}_ts{}_e{}_lr{:0.5f}_w{:0.5f}_bs{}_vocab{}_beam{}".format(args.dataset, attention_s, train_size, val_size, test_size, epoch, lr, weight_decay, batch_size, input_size, beam_size)
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
        attention,
        beam_size
    )


    # plot the result
    epochs = len(encoder_decoder_solver.log.keys())
    train_losses = [encoder_decoder_solver.log[i]["train_loss"] for i in range(epochs)]
    # val_losses = [encoder_decoder_solver.log[i]["val_loss"] for i in range(epochs)]train_perplexity
    train_blues_1 = [encoder_decoder_solver.log[i]["train_bleu_1"] for i in range(epochs)]
    train_blues_2 = [encoder_decoder_solver.log[i]["train_bleu_2"] for i in range(epochs)]
    train_blues_3 = [encoder_decoder_solver.log[i]["train_bleu_3"] for i in range(epochs)]
    train_blues_4 = [encoder_decoder_solver.log[i]["train_bleu_4"] for i in range(epochs)]
    val_blues_1 = [encoder_decoder_solver.log[i]["val_bleu_1"] for i in range(epochs)]
    val_blues_2 = [encoder_decoder_solver.log[i]["val_bleu_2"] for i in range(epochs)]
    val_blues_3 = [encoder_decoder_solver.log[i]["val_bleu_3"] for i in range(epochs)]
    val_blues_4 = [encoder_decoder_solver.log[i]["val_bleu_4"] for i in range(epochs)]
    train_cider = [encoder_decoder_solver.log[i]["train_cider"] for i in range(epochs)]
    val_cider = [encoder_decoder_solver.log[i]["val_cider"] for i in range(epochs)]
    # train_meteor = [encoder_decoder_solver.log[i]["train_meteor"] for i in range(epochs)]
    # val_meteor = [encoder_decoder_solver.log[i]["val_meteor"] for i in range(epochs)]
    train_rouge = [encoder_decoder_solver.log[i]["train_rouge"] for i in range(epochs)]
    val_rouge = [encoder_decoder_solver.log[i]["val_rouge"] for i in range(epochs)]

    # plot training curve
    print("plotting training curves...\n")
    if not os.path.exists("outputs/curves/{}".format(settings)):
        os.mkdir("outputs/curves/{}".format(settings))
    plt.switch_backend("agg")
    fig = plt.gcf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_losses, label="train_loss")
    # plt.plot(range(epochs), val_losses, label="val_loss")
    # plt.plot(range(epochs), train_perplexity, label="train_perplexity")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("outputs/curves/{}/loss_curve_{}.png".format(settings, settings), bbox_inches="tight")
    # plot the bleu scores
    fig.clf()
    fig.set_size_inches(16,32)
    plt.subplot(4, 1, 1)
    plt.plot(range(epochs), train_blues_1, "C3", label="train_bleu")
    plt.plot(range(epochs), val_blues_1, "C4", label="val_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-1')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(range(epochs), train_blues_2, "C3", label="train_bleu")
    plt.plot(range(epochs), val_blues_2, "C4", label="val_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-2')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(range(epochs), train_blues_3, "C3", label="train_bleu")
    plt.plot(range(epochs), val_blues_3, "C4", label="val_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-3')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(range(epochs), train_blues_4, "C3", label="train_bleu")
    plt.plot(range(epochs), val_blues_4, "C4", label="val_bleu")
    plt.xlabel('epoch')
    plt.ylabel('BLEU-4')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("outputs/curves/{}/bleu_curve_{}.png".format(settings, settings), bbox_inches="tight")
    # plot the cider scores
    fig.clf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_cider, label="train_cider")
    plt.plot(range(epochs), val_cider, label="val_cider")
    plt.xlabel('epoch')
    plt.ylabel('CIDEr')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("outputs/curves/{}/cider_curve_{}.png".format(settings, settings), bbox_inches="tight")
    # # plot the meteor scores
    # fig.clf()
    # fig.set_size_inches(16,8)
    # plt.plot(range(epochs), train_meteor, label="train_meteor")
    # plt.plot(range(epochs), val_meteor, label="val_meteor")
    # plt.xlabel('epoch')
    # plt.ylabel('METEOR')
    # plt.xticks(range(0, epochs + 1,  math.floor(epoch / 10)))
    # plt.legend()
    # plt.savefig("outputs/curves/meteor_curve_%s_ts%d_e%d_lr%f_bs%d_vocal%d.png" % (model_type, train_size, epoch, lr, batch_size, input_size), bbox_inches="tight")
    # plot the rouge scores
    fig.clf()
    fig.set_size_inches(16,8)
    plt.plot(range(epochs), train_rouge, label="train_rouge")
    plt.plot(range(epochs), val_rouge, label="val_rouge")
    plt.xlabel('epoch')
    plt.ylabel('ROUGE_L')
    plt.xticks(range(1, epochs + 1,  math.floor(epoch / 10)))
    plt.legend()
    plt.savefig("outputs/curves/{}/rouge_curve_{}.png".format(settings, settings), bbox_inches="tight")

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
        if os.path.exists("outputs/results/{}.json".format(settings)):
            print("loading existing results...")
            print()
            candidates = json.load(open("outputs/results/{}.json".format(settings)))
        else:
            print("evaluating with beam search...")
            print()
            for _, (model_ids, _, captions, embeddings, embeddings_interm, lengths) in enumerate(dataloader['test']):
                if attention:
                    visual_inputs = Variable(embeddings_interm, requires_grad=False).cuda()
                else:
                    visual_inputs = Variable(embeddings, requires_grad=False).cuda()
                caption_inputs = Variable(captions[:, :-1], requires_grad=False).cuda()
                cap_lengths = Variable(lengths, requires_grad=False).cuda()
                visual_contexts = encoder(visual_inputs)
                max_length = int(cap_lengths[0].item()) + 10
                for bs in beam_size:
                    if attention:
                        outputs[bs] = decoder.beam_search(visual_contexts, caption_inputs, int(bs), max_length)
                        outputs[bs] = decode_attention_outputs(outputs[bs], None, dict_idx2word, "val")
                    else:
                        outputs[bs] = decoder.beam_search(visual_contexts, int(bs), max_length)
                        outputs[bs] = decode_outputs(outputs[bs], None, dict_idx2word, "val")
                    
                    for model_id, output in zip(model_ids, outputs[bs]):
                        if model_id not in candidates[bs].keys():
                            candidates[bs][model_id] = [output]
                        else:
                            candidates[bs][model_id].append(output)
            # save results
            json.dump(candidates, open("outputs/results/{}.json".format(settings), 'w'))

        with open("outputs/scores/{}.txt".format(settings), 'w') as f:
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
                # write report
                f.write("----------------------Beam_size: {}-----------------------\n".format(bs))
                f.write("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[bs][0][0], max(bleu[bs][1][0]), min(bleu[bs][1][0])))
                f.write("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[bs][0][1], max(bleu[bs][1][1]), min(bleu[bs][1][1])))
                f.write("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[bs][0][2], max(bleu[bs][1][2]), min(bleu[bs][1][2])))
                f.write("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[bs][0][3], max(bleu[bs][1][3]), min(bleu[bs][1][3])))
                f.write("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(cider[bs][0], max(cider[bs][1]), min(cider[bs][1])))
                f.write("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n\n".format(rouge[bs][0], max(rouge[bs][1]), min(rouge[bs][1])))


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
    parser.add_argument("--attention", type=str, default='none', help="att2all/att2in/spatial/adaptive/none")
    parser.add_argument("--evaluation", type=str, default="false", help="true/false")
    args = parser.parse_args()
    main(args)
    