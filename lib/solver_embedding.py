import torch
import time
import math
import json
import os
import random
from datetime import datetime
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tensorboardX import SummaryWriter
import lib.configs as configs 
from torch.nn.utils import clip_grad_value_

class EmbeddingSolver():
    def __init__(self, criterion, optimizer, settings, reduce_step):
        self.criterion = criterion
        self.optimizer = optimizer
        self.settings = settings
        self.reduce_step = reduce_step

    def forward(self, shape_encoder, text_encoder, shapes, texts, labels):
        # load
        shapes = shapes.cuda().index_select(0, torch.LongTensor([i * 2 for i in range(shapes.size(0) // 2)]).cuda())
        texts = texts.cuda()
        shape_labels = labels.cuda().index_select(0, torch.LongTensor([i * 2 for i in range(labels.size(0) // 2)]).cuda())
        text_labels = labels.cuda()
        
        # forward pass
        s = shape_encoder(shapes)
        t = text_encoder(texts)
        
        # compute train_log
        walker_loss_tst = self.criterion['walker'](t, s, text_labels)
        walker_loss_sts = self.criterion['walker'](s, t, shape_labels)
        visit_loss_ts = self.criterion['visit'](t, s)
        visit_loss_st = self.criterion['visit'](s, t)

        # ML
        # TT
        embedding = t
        metric_loss_tt = self.criterion['metric'](embedding, 'TT')
        # ST
        s_mask = torch.ByteTensor([[1], [0]]).repeat(t.size(0) // 2, 128).cuda()
        t_mask = torch.ByteTensor([[0], [1]]).repeat(t.size(0) // 2, 128).cuda()
        selected_s = s
        selected_t = t.index_select(0, torch.LongTensor([i * 2 for i in range(t.size(0) // 2)]).cuda())
        masked_s = torch.zeros(t.size(0), 128).cuda().masked_scatter_(s_mask, selected_s)
        masked_t = torch.zeros(t.size(0), 128).cuda().masked_scatter_(t_mask, selected_t)
        embedding = masked_s + masked_t
        metric_loss_st = self.criterion['metric'](embedding, 'ST')
        # flip t
        flipped_t = t.index_select(0, torch.LongTensor([i * 2 + 1 for i in range(t.size(0) // 2)]).cuda())
        flipped_masked_t = torch.zeros(t.size(0), 128).cuda().masked_scatter_(t_mask, flipped_t)
        embedding = masked_s + flipped_masked_t
        metric_loss_st += self.criterion['metric'](embedding, 'ST')
        
        # add norm penalty
        shape_norm_penalty = self._norm_penalty(s)
        text_norm_penalty = self._norm_penalty(t)

        # accumulate loss
        loss = walker_loss_tst + walker_loss_sts + visit_loss_ts + visit_loss_st
        loss += configs.METRIC_MULTIPLIER * (metric_loss_st + metric_loss_tt)
        loss += (configs.SHAPE_NORM_MULTIPLIER * shape_norm_penalty + configs.TEXT_NORM_MULTIPLIER * text_norm_penalty)

        losses = {
            'loss': loss,
            'walker_loss_tst': walker_loss_tst,
            'walker_loss_sts': walker_loss_sts,
            'visit_loss_ts': visit_loss_ts,
            'visit_loss_st': visit_loss_st,
            'metric_loss_st': metric_loss_st,
            'metric_loss_tt': metric_loss_tt,
            'shape_norm_penalty': shape_norm_penalty,
            'text_norm_penalty': text_norm_penalty
        }

        return losses

    def evaluate(self, shape_encoder, text_encoder, dataloader, val_log):
        for _, (_, shapes, texts, _, labels) in enumerate(dataloader['val']):
            start = time.time()
            # forward pass
            losses = self.forward(shape_encoder, text_encoder, shapes, texts, labels)

            # record
            val_log['val_loss'].append(losses['loss'].item())
            val_log['walker_loss_tst'].append(losses['walker_loss_tst'].item())
            val_log['walker_loss_sts'].append(losses['walker_loss_sts'].item())
            val_log['visit_loss_ts'].append(losses['visit_loss_ts'].item())
            val_log['visit_loss_st'].append(losses['visit_loss_st'].item())
            val_log['metric_loss_st'].append(losses['metric_loss_st'].item())
            val_log['metric_loss_tt'].append(losses['metric_loss_tt'].item())
            val_log['shape_norm_penalty'].append(losses['shape_norm_penalty'].item())
            val_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())
            val_log['iter_time'].append(time.time() - start)

        return val_log

    def train(self, shape_encoder, text_encoder, rank, best, lock, dataloader, epoch, verbose):
        print("[{}] starting...\n".format(rank))
        total_iter = len(dataloader['train']) * epoch
        iter_count = 0
        scheduler = StepLR(self.optimizer, step_size=self.reduce_step, gamma=0.8)
        for epoch_id in range(epoch):
            print("[{}] epoch [{}/{}] starting...\n".format(rank, epoch_id+1, epoch))
            scheduler.step()
            train_log = {
                'forward': [],
                'backward': [],
                'iter_time': [],
                'train_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_st': [],
                'metric_loss_tt': [],
                'shape_norm_penalty': [],
                'text_norm_penalty': []
            }
            val_log = {
                'iter_time': [],
                'val_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_st': [],
                'metric_loss_tt': [],
                'shape_norm_penalty': [],
                'text_norm_penalty': []
                
            }
            for _, (_, shapes, texts, _, labels) in enumerate(dataloader['train']):
                start = time.time()
                # forward pass
                forward_since = time.time()
                losses = self.forward(shape_encoder, text_encoder, shapes, texts, labels)
                train_log['forward'].append(time.time() - forward_since)
                # record
                train_log['train_loss'].append(losses['loss'].item())
                train_log['walker_loss_tst'].append(losses['walker_loss_tst'].item())
                train_log['walker_loss_sts'].append(losses['walker_loss_sts'].item())
                train_log['visit_loss_ts'].append(losses['visit_loss_ts'].item())
                train_log['visit_loss_st'].append(losses['visit_loss_st'].item())
                train_log['metric_loss_st'].append(losses['metric_loss_st'].item())
                train_log['metric_loss_tt'].append(losses['metric_loss_tt'].item())
                train_log['shape_norm_penalty'].append(losses['shape_norm_penalty'].item())
                train_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())

                # back prop
                self.optimizer.zero_grad()
                backward_since = time.time()
                losses['loss'].backward()
                clip_grad_value_(list(shape_encoder.parameters()) + list(text_encoder.parameters()), configs.CLIP_VALUE)
                self.optimizer.step()
                train_log['backward'].append(time.time() - backward_since)

                # record
                train_log['iter_time'].append(time.time() - start)
                iter_count += 1

                # report
                if iter_count % verbose == 0:
                    self._iter_report(train_log, rank, iter_count, total_iter)

            # evaluate
            val_log = self.evaluate(shape_encoder, text_encoder, dataloader, val_log)
            
            # epoch report
            self._epoch_report(train_log, val_log, rank, epoch_id, epoch)
            
            # best
            with lock:
                if np.mean(val_log['val_loss']) < best['loss'].value:
                    # report best
                    print("[{}] best_loss achieved: {}".format(rank, np.mean(val_log['val_loss'])))
                    best['rank'].value = rank
                    best['epoch'].value = epoch_id
                    best['loss'].value = float(np.mean(val_log['val_loss']))
                    best['walker_loss_tst'].value = float(np.mean(val_log['walker_loss_tst']))
                    best['walker_loss_sts'].value = float(np.mean(val_log['walker_loss_sts']))
                    best['visit_loss_ts'].value = float(np.mean(val_log['visit_loss_ts']))
                    best['visit_loss_st'].value = float(np.mean(val_log['visit_loss_st']))
                    best['metric_loss_st'].value = float(np.mean(val_log['metric_loss_st']))
                    best['metric_loss_tt'].value = float(np.mean(val_log['metric_loss_tt']))
                    best['shape_norm_penalty'].value = float(np.mean(val_log['shape_norm_penalty']))
                    best['text_norm_penalty'].value = float(np.mean(val_log['text_norm_penalty']))

                    # save the best models
                    print("[{}] saving models...\n".format(rank))
                    if not os.path.exists(os.path.join(configs.OUTPUT_EMBEDDING, self.settings, "models")):
                        os.mkdir(os.path.join(configs.OUTPUT_EMBEDDING, self.settings, "models"))
                    torch.save(shape_encoder, os.path.join(configs.OUTPUT_EMBEDDING, self.settings, "models", "shape_encoder.pth"))
                    torch.save(text_encoder, os.path.join(configs.OUTPUT_EMBEDDING, self.settings, "models", "text_encoder.pth"))

        # done
        print("[{}] done...\n".format(rank))

        # return best['shape_encoder'], best['text_encoder']

    def _norm_penalty(self, embedding):
        '''
        added penalty to the loss if embedding's loss exceeds the threshold
        '''
        norm = torch.norm(embedding, p=2, dim=1)
        penalty = torch.max(torch.zeros(norm.size()).cuda(), norm - configs.MAX_NORM).mean()

        return penalty

    def _iter_report(self, log, rank, iter_count, total_iter):
        # compute ETA
        exetime_s = np.mean(log['iter_time'])
        eta_s = exetime_s * (total_iter - iter_count)
        eta_m = math.floor(eta_s / 60)
        
        # show report
        print("----------------------[{}]iter: [{}/{}]----------------------".format(rank, iter_count, total_iter))
        print("[loss] loss: %f" % (
            np.mean(log['train_loss'])
        ))
        print("[loss] walker_loss_tst: %f, walker_loss_sts: %f" % (
            np.mean(log['walker_loss_tst']),
            np.mean(log['walker_loss_sts'])
        ))
        print("[loss] visit_loss_ts: %f, visit_loss_st: %f" % (
            np.mean(log['visit_loss_ts']),
            np.mean(log['visit_loss_st'])
        ))
        print("[loss] metric_loss_st: %f, metric_loss_tt: %f" % (
            np.mean(log['metric_loss_st']),
            np.mean(log['metric_loss_tt'])
        ))
        print("[loss] shape_norm_penalty: %f, text_norm_penalty: %f" % (
            np.mean(log['shape_norm_penalty']),
            np.mean(log['text_norm_penalty'])
        ))
        print("[info] forward_per_iter: %fs\n[info] backward_per_iter: %fs" % ( 
            np.mean(log['forward']),
            np.mean(log['backward'])
        ))
        print("[info] time_per_iter: %fs\n[info] ETA: %dm %ds\n" % ( 
            np.mean(log['iter_time']),
            eta_m,
            eta_s - eta_m * 60
        ))

    def _epoch_report(self, train_log, val_log, rank, epoch_id, epoch):
        # show report
        print("[{}] epoch [{}/{}] done...".format(rank, epoch_id+1, epoch))
        print("------------------------summary------------------------")
        print("[train] loss: %f" % (
            np.mean(train_log['train_loss'])
        ))
        print("[val]   loss: %f" % (
            np.mean(val_log['val_loss'])
        ))
        print("[train] walker_loss_tst: %f, walker_loss_sts: %f" % (
            np.mean(train_log['walker_loss_tst']),
            np.mean(train_log['walker_loss_sts'])
        ))
        print("[val]   walker_loss_tst: %f, walker_loss_sts: %f" % (
            np.mean(val_log['walker_loss_tst']),
            np.mean(val_log['walker_loss_sts'])
        ))
        print("[train] visit_loss_ts: %f, visit_loss_st: %f" % (
            np.mean(train_log['visit_loss_ts']),
            np.mean(train_log['visit_loss_st'])
        ))
        print("[val]   visit_loss_ts: %f, visit_loss_st: %f" % (
            np.mean(val_log['visit_loss_ts']),
            np.mean(val_log['visit_loss_st'])
        ))
        print("[train] metric_loss_st: %f, metric_loss_tt: %f" % (
            np.mean(train_log['metric_loss_st']),
            np.mean(train_log['metric_loss_tt'])
        ))
        print("[val]   metric_loss_st: %f, metric_loss_tt: %f" % (
            np.mean(val_log['metric_loss_st']),
            np.mean(val_log['metric_loss_tt'])
        ))
        print("[train] shape_norm_penalty: %f, text_norm_penalty: %f" % (
            np.mean(train_log['shape_norm_penalty']),
            np.mean(train_log['text_norm_penalty'])
        ))
        print("[val]   shape_norm_penalty: %f, text_norm_penalty: %f\n" % (
            np.mean(val_log['shape_norm_penalty']),
            np.mean(val_log['text_norm_penalty'])
        ))
