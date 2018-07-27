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

    def train(self, shape_encoder, text_encoder, rank, best_loss, lock, dataloader, epoch, verbose):
        log = {
            'forward': [],
            'backward': [],
            'iter_time': []
        }
        best = {
            'train_loss': np.inf,
            'walker_loss_tst': np.inf,
            'walker_loss_sts': np.inf,
            'visit_loss_ts': np.inf,
            'visit_loss_st': np.inf,
            'metric_loss_ss': np.inf,
            'metric_loss_st': np.inf,
            'shape_encoder': None,
            'text_encoder': None
        }
        total_iter = len(dataloader) * epoch
        iter_count = 0
        scheduler = StepLR(self.optimizer, step_size=self.reduce_step, gamma=0.8)
        for epoch_id in range(epoch):
            scheduler.step()
            loss = {
                'train_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_ss': [],
                'metric_loss_st': [],
                
            }
            for iter_id, (_, shapes, texts, _, labels) in enumerate(dataloader):
                start = time.time()
                # load
                shapes = shapes.cuda()
                texts = texts.cuda()
                labels = labels.cuda()
                
                # forward pass
                forward_since = time.time()
                s = shape_encoder(shapes)
                t = text_encoder(texts)
                
                # compute loss
                walker_loss_tst = self.criterion['walker_tst'](t, s, labels)
                walker_loss_sts = self.criterion['walker_sts'](s, t, labels)
                visit_loss_ts = self.criterion['visit_ts'](t, s, labels)
                visit_loss_st = self.criterion['visit_st'](s, t, labels)
                metric_loss_ss = self.criterion['metric_ss'](s, s, labels)
                metric_loss_st = self.criterion['metric_st'](s, t, labels)

                # rescale the loss
                walker_loss_tst[walker_loss_tst == float("Inf") or walker_loss_tst == float("Nan")] = 0.
                walker_loss_sts[walker_loss_sts == float("Inf") or walker_loss_sts == float("Nan")] = 0.
                visit_loss_ts[visit_loss_ts == float("Inf") or visit_loss_ts == float("Nan")] = 0.
                visit_loss_st[visit_loss_st == float("Inf") or visit_loss_st == float("Nan")] = 0.
                metric_loss_ss[metric_loss_ss == float("Inf") or metric_loss_ss == float("Nan")] = 0.
                metric_loss_st[metric_loss_st == float("Inf") or metric_loss_st == float("Nan")] = 0.

                # accumulate loss
                train_loss = walker_loss_tst + walker_loss_sts + visit_loss_ts + visit_loss_st + configs.METRIC_MULTIPLIER * metric_loss_ss + 2. * configs.METRIC_MULTIPLIER * metric_loss_st
                log['forward'].append(time.time() - forward_since)

                # back prop
                self.optimizer.zero_grad()
                backward_since = time.time()
                train_loss.backward()
                clip_grad_value_(list(shape_encoder.parameters()) + list(text_encoder.parameters()), configs.CLIP_VALUE)
                self.optimizer.step()
                log['backward'].append(time.time() - backward_since)

                # record
                loss['walker_loss_tst'].append(walker_loss_tst.item())
                loss['walker_loss_sts'].append(walker_loss_sts.item())
                loss['visit_loss_ts'].append(visit_loss_ts.item())
                loss['visit_loss_st'].append(visit_loss_st.item())
                loss['metric_loss_ss'].append(metric_loss_ss.item())
                loss['metric_loss_st'].append(metric_loss_st.item())
                loss['train_loss'].append(train_loss.item())
                iter_count += 1

                # compute ETA
                log['iter_time'].append(time.time() - start)
                exetime_s = np.mean(log['iter_time'])
                eta_s = exetime_s * (total_iter - iter_count)
                eta_m = math.floor(eta_s / 60)
                
                # report
                if iter_count % verbose == 0:
                    print("------------------------[{}]Iter: [{}/{}]------------------------".format(rank, iter_count, total_iter))
                    print("[Loss] train_loss: %f" % (
                        np.mean(loss['train_loss'])
                    ))
                    print("[Loss] walker_loss_tst: %f, walker_loss_sts: %f" % (
                        np.mean(loss['walker_loss_tst']),
                        np.mean(loss['walker_loss_sts'])
                    ))
                    print("[Loss] visit_loss_ts: %f, visit_loss_st: %f" % (
                        np.mean(loss['visit_loss_ts']),
                        np.mean(loss['visit_loss_st'])
                    ))
                    print("[Loss] metric_loss_ss: %f, metric_loss_st: %f" % (
                        np.mean(loss['metric_loss_ss']),
                        np.mean(loss['metric_loss_st'])
                    ))
                    print("[Info] forward: %fs\n[Info] backward: %fs" % ( 
                        np.mean(log['forward']),
                        np.mean(log['backward'])
                    ))
                    print("[Info] time_per_iter: %fs\n[Info] ETA: %dm %ds\n\n" % ( 
                        np.mean(log['iter_time']),
                        eta_m,
                        eta_s - eta_m * 60
                    ))
                
            # best
            with lock:
                if np.mean(loss['train_loss']) < best_loss:
                    best_loss.value = np.mean(loss['train_loss'])
                    best['train_loss'] = np.mean(loss['train_loss'])
                    best['walker_loss_tst'] = np.mean(loss['walker_loss_tst'])
                    best['walker_loss_sts'] = np.mean(loss['walker_loss_sts'])
                    best['visit_loss_ts'] = np.mean(loss['visit_loss_ts'])
                    best['visit_loss_st'] = np.mean(loss['visit_loss_st'])
                    best['metric_loss_ss'] = np.mean(loss['metric_loss_ss'])
                    best['metric_loss_st'] = np.mean(loss['metric_loss_st'])

                    # save the best models
                    print("saving models...")
                    torch.save(shape_encoder, "outputs/models/embeddings/shape_encoder_{}.pth".format(self.settings))
                    torch.save(text_encoder, "outputs/models/embeddings/text_encoder_{}.pth".format(self.settings))

        # report best
        print("------------------------best------------------------")
        print("[Loss] train_loss: %f" % (
            best['train_loss']
        )), 
        print("[Loss] walker_loss_tst: %f, walker_loss_sts: %f" % (
            best['walker_loss_tst'],
            best['walker_loss_sts']
        ))
        print("[Loss] visit_loss_ts: %f, visit_loss_st: %f" % (
            best['visit_loss_ts'],
            best['visit_loss_st']
        ))
        print("[Loss] metric_loss_ss: %f, metric_loss_st: %f\n" % (
            best['metric_loss_ss'],
            best['metric_loss_st']
        ))

        # return best['shape_encoder'], best['text_encoder']
