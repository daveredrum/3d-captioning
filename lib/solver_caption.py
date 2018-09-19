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
import lib.capeval.bleu.bleu as capbleu
import lib.capeval.cider.cider as capcider
import lib.capeval.meteor.meteor as capmeteor
import lib.capeval.rouge.rouge as caprouge
from lib.utils import *
from lib.configs import CONF

class EncoderDecoderSolver():
    def __init__(self, optimizer, criterion, output_root):
        self.optimizer = optimizer
        self.criterion = criterion
        self.output_root = output_root
        self.threshold = {
            'schedule': CONF.CAP.SCHEDULE_THRESHOLD, 
            'save': CONF.CAP.SAVE_THRESHOLD
        }
        self.log = {}

    def train(self, encoder, decoder, dataloader, references, dict_word2idx, dict_idx2word, epoch, verbose, attention, beam_size=1):
        scheduler = ReduceLROnPlateau(self.optimizer, factor=0.8, patience=5, threshold=0.001)
        # scheduler = StepLR(self.optimizer, gamma=0.8, step_size=3)
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
            'encoder': None,
            'decoder': None,
        }
        for epoch_id in range(epoch):
            print("epoch [{}/{}] starting...\n".format(epoch_id + 1, epoch))
            # scheduler.step()
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
                for iter_id, (model_ids, captions, embeddings, embeddings_interm, lengths) in enumerate(dataloader[phase]):
                    # decoder without attention
                    if attention == "fc":
                        visual_inputs = embeddings.cuda()
                        caption_inputs = captions[:, :-1].cuda()
                        caption_targets = captions.cuda()
                        cap_lengths = lengths.cuda()
                        
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
                            if verbose and (iter_id+1) % verbose == 0:
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

                    else:
                        visual_inputs = embeddings_interm.cuda()
                        caption_inputs = captions[:, :-1].cuda()
                        caption_targets = captions[:, 1:].cuda() 
                        cap_lengths = lengths.cuda()
                        
                        if phase == "train":
                            encoder.train()
                            decoder.train()
                            self.optimizer.zero_grad()
                            # forward pass
                            forward_since = time.time()
                            visual_contexts = encoder(visual_inputs)
                            # visual_contexts = (batch_size, visual_channels, visual_size, visual_size)
                            # teacher forcing
                            states = decoder.init_hidden(visual_contexts[0])
                            outputs = decoder(visual_contexts, caption_inputs, states)

                            outputs_packed = pack_padded_sequence(outputs, [l-1 for l in cap_lengths], batch_first=True)[0]
                            targets = pack_padded_sequence(caption_targets, [l-1 for l in cap_lengths], batch_first=True)[0]
                            loss = self.criterion(outputs_packed, targets)

                            # decode outputs
                            outputs = decode_attention_outputs(outputs, cap_lengths, dict_idx2word, phase)
                            
                            # save to candidates
                            for model_id, output in zip(model_ids, outputs):
                                if model_id not in candidates[phase].keys():
                                    candidates[phase][model_id] = [output]
                                else:
                                    candidates[phase][model_id].append(output)
                            
                            log['forward'].append(time.time() - forward_since)
                            # backward pass
                            backward_since = time.time()
                            loss.backward()
                            # clipping the gradient
                            clip_grad_value_(self.optimizer, 5)
                            self.optimizer.step()
                            log['backward'].append(time.time() - backward_since)
                            log['train_loss'].append(loss.item())
                            log['train_perplexity'].append(np.exp(loss.item()))

                            # report
                            if verbose and (iter_id+1) % verbose == 0:
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
                            max_length = int(cap_lengths[0].item()) + 10
                            outputs = decoder.beam_search(visual_contexts, caption_inputs, beam_size, max_length)

                            # decode the outputs
                            outputs = decode_attention_outputs(outputs, None, dict_idx2word, phase)
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
            if log['train_loss'] <= self.threshold['schedule']:
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
            print("----------------------summary [{}/{}]-----------------------".format(epoch_id+1, epoch))
            print("[Loss]    train_loss: %f, perplexity: %f" % (
                log['train_loss'], 
                log['train_perplexity'])
            )
            print("[BLEU-1]  train_bleu: %f, val_bleu: %f" % (
                log['train_bleu_1'],
                log['val_bleu_1'])
            )
            print("[BLEU-2]  train_bleu: %f, val_bleu: %f" % (
                log['train_bleu_2'],
                log['val_bleu_2'])
            )
            print("[BLEU-3]  train_bleu: %f, val_bleu: %f" % (
                log['train_bleu_3'],
                log['val_bleu_3'])
            )
            print("[BLEU-4]  train_bleu: %f, val_bleu: %f" % (
                log['train_bleu_4'],
                log['val_bleu_4'])
            )
            print("[CIDEr]   train_cider: %f, val_cider: %f" % (
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
            print("[Info]    forward_per_epoch: %fs\n[Info]    backward_per_epoch: %fs\n[Info]    val_per_epoch: %fs" % (
                np.sum(log['forward']), 
                np.sum(log['backward']),
                np.sum(log['val_time']))
            )
            print("[Info]    eval_time: %fs" % ( 
                np.mean(log['eval_time']))
            )
            print("[Info]    time_per_epoch: %fs\n[Info]    ETA: %dm %ds\n\n" % ( 
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
            if log['train_loss'] <= self.threshold['save'] and log['val_cider'] > best_scores["cider"]:
                print("best cider achieved:", log['val_cider'])
                print("current loss:", log['train_loss'])
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

                print("saving the best models...\n")
                model_root = os.path.join(self.output_root, "models")
                if not os.path.exists(model_root):
                    os.mkdir(model_root)
                torch.save(best_models['encoder'], os.path.join(model_root, "encoder.pth"))
                torch.save(best_models['decoder'], os.path.join(model_root, "decoder.pth"))

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
        if not best_models['encoder'] or not best_models['decoder']:
            best_models['encoder'] = encoder
            best_models['decoder'] = decoder

        return best_models['encoder'], best_models['decoder']
                