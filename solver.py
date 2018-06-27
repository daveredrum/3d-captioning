import torch
import time
import math
import json
import os
import random
from datetime import datetime
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from tensorboardX import SummaryWriter
import capeval.bleu.bleu as capbleu
import capeval.cider.cider as capcider
import capeval.meteor.meteor as capmeteor
import capeval.rouge.rouge as caprouge
from utils import *

class EncoderDecoderSolver():
    def __init__(self, optimizer, criterion, settings):
        self.optimizer = optimizer
        self.criterion = criterion
        self.settings = settings
        self.log = {}

    def train(self, encoder, decoder, dataloader, references, dict_word2idx, dict_idx2word, epoch, verbose, beam_size=3):
        scheduler = ReduceLROnPlateau(self.optimizer, factor=0.8, patience=5, threshold=0.001)
        best_info = {
            'epoch_id': 0,
            'loss': 0,
        }
        best_scores = {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0,
            'cider': 0.0,
            'rouge': 0.0,
        }
        best_models = {
            'encoder': encoder,
            'decoder': decoder,
        }
        for epoch_id in range(epoch):
            print("---------------------epoch %d/%d----------------------" % (epoch_id + 1, epoch))
            log = {
                'train_loss': [],
                'train_perplexity': [],
                'train_bleu_1': [],
                'train_bleu_2': [],
                'train_bleu_3': [],
                'train_bleu_4': [],
                'val_loss': [],
                'val_bleu_1': [],
                'val_bleu_2': [],
                'val_bleu_3': [],
                'val_bleu_4': [],
                'train_cider': [],
                'val_cider': [],
                # 'train_meteor': [],
                # 'val_meteor': [],
                'train_rouge': [],
                'val_rouge': [],
                'forward': [],
                'backward': [],
                'val_time': [],
                'eval_time': [],
                'epoch_time': []
            }
            candidates = {
                'train': {},
                'val': {}
            }
            start = time.time()
            for phase in ["train", "val"]:
                total_iter = len(dataloader[phase])
                for iter_id, (model_ids, captions, embeddings, lengths) in enumerate(dataloader[phase]):
                    # decoder without attention
                    visual_inputs = Variable(embeddings).cuda()
                    caption_inputs = Variable(captions[:, :-1]).cuda()
                    caption_targets = Variable(captions).cuda()
                    cap_lengths = Variable(lengths).cuda()
                    
                    if phase == "train":
                        encoder.train()
                        decoder.train()
                        self.optimizer.zero_grad()
                        # forward pass
                        forward_since = time.time()
                        visual_contexts = encoder(visual_inputs)
                        # teacher forcing
                        states = decoder.init_hidden(visual_contexts)
                        outputs = decoder(visual_contexts, caption_inputs, states)
                        # # no teacher forcing
                        # outputs = decoder.sample(visual_contexts, cap_lengths)
                        outputs_packed = pack_padded_sequence(outputs, [l for l in cap_lengths], batch_first=True)[0]
                        targets = pack_padded_sequence(caption_targets, [l for l in cap_lengths], batch_first=True)[0]
                        loss = self.criterion(outputs_packed, targets)
                        
                        # decode outputs
                        outputs = decode_outputs(outputs, cap_lengths, dict_idx2word, phase)
                        # save to candidates
                        for model_id, output in zip(model_ids, outputs):
                            if model_id not in candidates[phase].keys():
                                candidates[phase][model_id] = [output]
                            else:
                                candidates[phase][model_id].append(output)
                        log['forward'].append(time.time() - forward_since)

                        # backward pass
                        # save log
                        backward_since = time.time()
                        loss.backward()
                        # clipping the gradient
                        clip_grad_value_(self.optimizer, 5)
                        self.optimizer.step()
                        log['backward'].append(time.time() - backward_since)
                        log['train_loss'].append(loss.item())
                        log['train_perplexity'].append(np.exp(loss.item()))

                        # report
                        if (iter_id+1) % verbose == 0:
                            print("Epoch: [{}/{}] Iter: [{}/{}] train_loss: {:.4f} perplexity: {:.4f}".format(
                                epoch_id+1,
                                epoch,
                                iter_id+1, 
                                total_iter, 
                                log['train_loss'][-1], 
                                log['train_perplexity'][-1]
                            ))
                    else:
                        # validate
                        encoder.eval()
                        decoder.eval()
                        val_since = time.time()
                        visual_contexts = encoder(visual_inputs)
                        # if beam_size == 1:
                        #     # greedy search
                        #     # generate until <END> token
                        #     outputs = []
                        #     states = decoder.init_hidden(visual_contexts)
                        #     max_length = int(cap_lengths[0].item()) + 10
                        #     for idx in range(visual_contexts.size(0)):
                        #         h, c = states[0][idx].unsqueeze(0), states[1][idx].unsqueeze(0)
                        #         temp = []
                        #         for i in range(max_length):
                        #             if i == 0:
                        #                 embedded = visual_contexts[idx].unsqueeze(0)
                        #                 predicted, (h, c) = decoder.sample(embedded, (h, c))
                        #                 inputs = caption_inputs[idx, 0].view(1)
                        #                 temp.append(predicted.max(2)[1].view(1).item())
                        #             else:
                        #                 embedded = decoder.embedding(inputs)
                        #                 predicted, (h, c) = decoder.sample(embedded, (h, c))
                        #                 inputs = predicted.max(2)[1].view(1)
                        #                 temp.append(inputs[0].item())
                        #             if inputs[0].item() == dict_word2idx['<END>']:
                        #                 break
                        #         outputs.append(temp)
                        # else:
                        #     # beam search
                        #     max_length = int(cap_lengths[0].item()) + 10
                        #     outputs = decoder.beam_search(visual_contexts, beam_size, max_length)
                        # beam search
                        max_length = int(cap_lengths[0].item()) + 10
                        outputs = decoder.beam_search(visual_contexts, beam_size, max_length)

                        # decode outputs
                        outputs = decode_outputs(outputs, None, dict_idx2word, phase)
                        # save to candidates
                        for model_id, output in zip(model_ids, outputs):
                            if model_id not in candidates[phase].keys():
                                candidates[phase][model_id] = [output]
                            else:
                                candidates[phase][model_id].append(output)

                        # save log
                        log['val_time'].append(time.time() - val_since)
                        

            # accumulate loss
            log['train_loss'] = np.mean(log['train_loss'])
            # log['val_loss'] = np.mean(log['val_loss'])
            log['train_perplexity'] = np.mean(log['train_perplexity'])
            # evaluate bleu
            eval_since = time.time()
            train_bleu, _ = capbleu.Bleu(4).compute_score(references["train"], candidates["train"])
            val_bleu, _ = capbleu.Bleu(4).compute_score(references["val"], candidates["val"])
            # evaluate cider
            train_cider, _ = capcider.Cider().compute_score(references["train"], candidates["train"])
            val_cider, _ = capcider.Cider().compute_score(references["val"], candidates["val"])
            # reduce the learning rate on plateau if training loss if training loss is small
            if log['train_loss'] <= 2.0:
                scheduler.step(val_cider)
            # # evaluate meteor
            # try:
            #     train_meteor, _ = capmeteor.Meteor().compute_score(references["train"], candidates["train"])
            #     val_meteor, _ = capmeteor.Meteor().compute_score(references["val"], candidates["val"])
            # except Exception:
            #     train_meteor = 0
            #     val_meteor = 0
            # evaluate rouge
            train_rouge, _ = caprouge.Rouge().compute_score(references["train"], candidates["train"])
            val_rouge, _ = caprouge.Rouge().compute_score(references["val"], candidates["val"])
            log['eval_time'] = time.time() - eval_since
            
            # log
            log['train_bleu_1'] = train_bleu[0]
            log['train_bleu_2'] = train_bleu[1]
            log['train_bleu_3'] = train_bleu[2]
            log['train_bleu_4'] = train_bleu[3]
            log['val_bleu_1'] = val_bleu[0]
            log['val_bleu_2'] = val_bleu[1]
            log['val_bleu_3'] = val_bleu[2]
            log['val_bleu_4'] = val_bleu[3]
            log['train_cider'] = train_cider
            log['val_cider'] = val_cider
            # log['train_meteor'] = train_meteor
            # log['val_meteor'] = val_meteor
            log['train_rouge'] = train_rouge
            log['val_rouge'] = val_rouge
            log['epoch_time'].append(np.mean(time.time() - start))
            # show report
            exetime_s = np.sum(log['epoch_time'])
            eta_s = exetime_s * (epoch - (epoch_id + 1))
            eta_m = math.floor(eta_s / 60)
            print("----------------------summary-----------------------")
            print("[Loss] train_loss: %f, perplexity: %f" % (
                log['train_loss'], 
                log['train_perplexity'])
            )
            print("[BLEU-1] train_bleu: %f, val_bleu: %f" % (
                log['train_bleu_1'],
                log['val_bleu_1'])
            )
            print("[BLEU-2] train_bleu: %f, val_bleu: %f" % (
                log['train_bleu_2'],
                log['val_bleu_2'])
            )
            print("[BLEU-3] train_bleu: %f, val_bleu: %f" % (
                log['train_bleu_3'],
                log['val_bleu_3'])
            )
            print("[BLEU-4] train_bleu: %f, val_bleu: %f" % (
                log['train_bleu_4'],
                log['val_bleu_4'])
            )
            print("[CIDEr] train_cider: %f, val_cider: %f" % (
                log['train_cider'],
                log['val_cider'])
            )
            # print("[METEOR] train_meteor: %f, val_meteor: %f" % (
            #     log['train_meteor'],
            #     log['val_meteor'])
            # )
            print("[ROUGE_L] train_rouge: %f, val_rouge: %f" % (
                log['train_rouge'],
                log['val_rouge'])
            )
            print("[Info]  forward_per_epoch: %fs\n[Info]  backward_per_epoch: %fs\n[Info]  val_per_epoch: %fs" % (
                np.sum(log['forward']), 
                np.sum(log['backward']),
                np.sum(log['val_time']))
            )
            print("[Info]  eval_time: %fs" % ( 
                np.mean(log['eval_time']))
            )
            print("[Info]  time_per_epoch: %fs\n[Info]  ETA: %dm %ds\n\n" % ( 
                np.mean(log['epoch_time']),
                eta_m,
                eta_s - eta_m * 60)
            )
            # print("[Debug] train_id: {}\n[Debug] train_ref: {}\n[Debug] train_can: {}\n".format(
            #     list(references["train"].keys())[0],
            #     references["train"][list(references["train"].keys())[0]],
            #     candidates["train"][list(references["train"].keys())[0]]
            # ))
            # print("[Debug] val_id: {}\n[Debug] val_ref: {}\n[Debug] val_can: {}\n\n".format(
            #     list(references["val"].keys())[0],
            #     references["val"][list(references["val"].keys())[0]],
            #     candidates["val"][list(references["val"].keys())[0]]
            # ))
            
            # save log
            self.log[epoch_id] = log
            
            # best
            if log['train_loss'] <= 2.0 and log['val_cider'] > best_scores["cider"]:
                best_info['epoch_id'] = epoch_id + 1
                best_info['loss'] = log['train_loss']
                best_scores['bleu_1'] = log['val_bleu_1']
                best_scores['bleu_2'] = log['val_bleu_2']
                best_scores['bleu_3'] = log['val_bleu_3']
                best_scores['bleu_4'] = log['val_bleu_4']
                best_scores['cider'] = log['val_cider']
                best_scores['rouge'] = log['val_rouge']
                best_models['encoder'] = encoder
                best_models['decoder'] = decoder

        # show the best
        print("---------------------best----------------------")
        print("[Best] Epoch_id: {}".format(best_info['epoch_id']))
        print("[Best] Loss: {}".format(best_info['loss']))
        print("[Best] BLEU-1: {}".format(best_scores['bleu_1']))
        print("[Best] BLEU-2: {}".format(best_scores['bleu_2']))
        print("[Best] BLEU-3: {}".format(best_scores['bleu_3']))
        print("[Best] BLEU-4: {}".format(best_scores['bleu_4']))
        print("[Best] CIDEr: {}".format(best_scores['cider']))
        print("[Best] ROUGE_L: {}".format(best_scores['rouge']))
        print()

        # save the best model
        print("saving the best models...\n")
        torch.save(best_models['encoder'], "models/encoder_{}.pth".format(self.settings))
        torch.save(best_models['decoder'], "models/decoder_{}.pth".format(self.settings))

        # save the best scores
        if not os.path.exists("scores/"):
            os.mkdir("scores/")
        json.dump(best_scores, open("scores/{}".format(self.settings), "w"))

        return best_models['encoder'], best_models['decoder']
                