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

    def evaluate(self, shape_encoder, text_encoder, dataloader):
        val = {
            'iter_time': [],
            'val_loss': [],
            'walker_loss_tst': [],
            'walker_loss_sts': [],
            'visit_loss_ts': [],
            'visit_loss_st': [],
            'metric_loss_tt': [],
            'metric_loss_ts': [],
            
        }
        for iter_id, (_, shapes, texts, _, labels) in enumerate(dataloader['val']):
            start = time.time()
            # load
            shapes = shapes.cuda()
            texts = texts.cuda()
            labels = labels.cuda()
            
            # forward pass
            s = shape_encoder(shapes)
            t = text_encoder(texts)

            # compute loss
            walker_loss_tst = self.criterion['walker_tst'](t, s, labels)
            walker_loss_sts = self.criterion['walker_sts'](s, t, labels)
            visit_loss_ts = self.criterion['visit_ts'](t, s, labels)
            visit_loss_st = self.criterion['visit_st'](s, t, labels)
            metric_loss_tt = self.criterion['metric_tt'](t, t, labels)
            metric_loss_ts = self.criterion['metric_ts'](t, s, labels)

            # accumulate loss
            val_loss = walker_loss_tst + walker_loss_sts + visit_loss_ts + visit_loss_st + configs.METRIC_MULTIPLIER * metric_loss_tt + 2 * configs.METRIC_MULTIPLIER * metric_loss_ts

            # record
            val['iter_time'].append(time.time() - start)
            val['val_loss'].append(val_loss.item())
            val['walker_loss_tst'].append(walker_loss_tst.item())
            val['walker_loss_sts'].append(walker_loss_sts.item())
            val['visit_loss_ts'].append(visit_loss_ts.item())
            val['visit_loss_st'].append(visit_loss_st.item())
            val['metric_loss_tt'].append(metric_loss_tt.item())
            val['metric_loss_ts'].append(metric_loss_ts.item())

        return val

    def train(self, shape_encoder, text_encoder, rank, best, lock, dataloader, epoch, verbose):
        print("[{}] starting...\n".format(rank))
        log = {
            'forward': [],
            'backward': [],
            'iter_time': []
        }
        total_iter = len(dataloader['train']) * epoch
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
                'metric_loss_tt': [],
                'metric_loss_ts': [],
                
            }
            for iter_id, (_, shapes, texts, _, labels) in enumerate(dataloader['train']):
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
                metric_loss_tt = self.criterion['metric_tt'](t, t, labels)
                metric_loss_ts = self.criterion['metric_ts'](t, s, labels)

                # accumulate loss
                train_loss = walker_loss_tst + walker_loss_sts + visit_loss_ts + visit_loss_st + configs.METRIC_MULTIPLIER * metric_loss_tt + 2 * configs.METRIC_MULTIPLIER * metric_loss_ts
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
                loss['metric_loss_tt'].append(metric_loss_tt.item())
                loss['metric_loss_ts'].append(metric_loss_ts.item())
                loss['train_loss'].append(train_loss.item())
                iter_count += 1

                # compute ETA
                log['iter_time'].append(time.time() - start)
                exetime_s = np.mean(log['iter_time'])
                eta_s = exetime_s * (total_iter - iter_count)
                eta_m = math.floor(eta_s / 60)
                
                # evaluate
                val = self.evaluate(shape_encoder, text_encoder, dataloader)

                # report
                if iter_count % verbose == 0:
                    print("------------------------[{}]Iter: [{}/{}]------------------------".format(rank, iter_count, total_iter))
                    print("[train] loss: %f" % (
                        np.mean(loss['train_loss'])
                    ))
                    print("[train] walker_loss_tst: %f, walker_loss_sts: %f" % (
                        np.mean(loss['walker_loss_tst']),
                        np.mean(loss['walker_loss_sts'])
                    ))
                    print("[train] visit_loss_ts: %f, visit_loss_st: %f" % (
                        np.mean(loss['visit_loss_ts']),
                        np.mean(loss['visit_loss_st'])
                    ))
                    print("[train] metric_loss_tt: %f, metric_loss_ts: %f" % (
                        np.mean(loss['metric_loss_tt']),
                        np.mean(loss['metric_loss_ts'])
                    ))
                    print("[val] loss: %f" % (
                        np.mean(val['val_loss'])
                    ))
                    print("[val] walker_loss_tst: %f, walker_loss_sts: %f" % (
                        np.mean(val['walker_loss_tst']),
                        np.mean(val['walker_loss_sts'])
                    ))
                    print("[val] visit_loss_ts: %f, visit_loss_st: %f" % (
                        np.mean(val['visit_loss_ts']),
                        np.mean(val['visit_loss_st'])
                    ))
                    print("[val] metric_loss_tt: %f, metric_loss_ts: %f" % (
                        np.mean(val['metric_loss_tt']),
                        np.mean(val['metric_loss_ts'])
                    ))
                    print("[Info] forward: %fs\n[Info] backward: %fs\n[Info] evaluate: %fs" % ( 
                        np.mean(log['forward']),
                        np.mean(log['backward']),
                        np.sum(val['iter_time'])
                    ))
                    print("[Info] time_per_iter: %fs\n[Info] ETA: %dm %ds\n" % ( 
                        np.mean(log['iter_time']),
                        eta_m,
                        eta_s - eta_m * 60
                    ))
                
            # best
            with lock:
                if np.mean(val['val_loss']) < best['loss'].value:
                    best['rank'].value = rank
                    best['epoch'].value = epoch_id
                    best['loss'].value = float(np.mean(val['val_loss']))
                    best['walker_loss_tst'].value = float(np.mean(val['walker_loss_tst']))
                    best['walker_loss_sts'].value = float(np.mean(val['walker_loss_sts']))
                    best['visit_loss_ts'].value = float(np.mean(val['visit_loss_ts']))
                    best['visit_loss_st'].value = float(np.mean(val['visit_loss_st']))
                    best['metric_loss_tt'].value = float(np.mean(val['metric_loss_tt']))
                    best['metric_loss_ts'].value = float(np.mean(val['metric_loss_ts']))

                    # save the best models
                    print("[{}] best_loss achieved: {}".format(rank, best['loss'].value))
                    print("[{}] saving models...\n".format(rank))
                    torch.save(shape_encoder, "outputs/models/embeddings/shape_encoder_{}.pth".format(self.settings))
                    torch.save(text_encoder, "outputs/models/embeddings/text_encoder_{}.pth".format(self.settings))

        # done
        print("[{}] done...\n".format(rank))

        # return best['shape_encoder'], best['text_encoder']
