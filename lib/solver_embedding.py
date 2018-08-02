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

    def evaluate(self, shape_encoder, text_encoder, dataloader, val_log):
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

            # ML
            # TT
            embedding = t
            metric_loss_tt = self.criterion['metric_tt'](embedding)
            # ST
            s_mask = torch.ByteTensor([[1], [0]]).repeat(t.size(0) // 2, 128).cuda()
            t_mask = torch.ByteTensor([[0], [1]]).repeat(t.size(0) // 2, 128).cuda()
            masked_s = torch.zeros(t.size(0), 128).cuda().masked_scatter(s_mask, s)
            masked_t = torch.zeros(t.size(0), 128).cuda().masked_scatter(t_mask, t)
            embedding = masked_s + masked_t
            metric_loss_st = self.criterion['metric_st'](embedding)
            # flip t
            inverted_t = t.index_select(0, torch.LongTensor([i * 2 for i in range(t.size(0) // 2)]).cuda())
            inverted_masked_t = torch.zeros(t.size(0), 128).cuda().masked_scatter(t_mask, inverted_t)
            embedding = masked_s + inverted_masked_t
            metric_loss_st += self.criterion['metric_st'](embedding)

            # accumulate loss
            val_loss = walker_loss_tst + walker_loss_sts + visit_loss_ts + visit_loss_st + configs.METRIC_MULTIPLIER * metric_loss_st + configs.METRIC_MULTIPLIER * metric_loss_tt

            # record
            val_log['iter_time'].append(time.time() - start)
            val_log['val_loss'].append(val_loss.item())
            val_log['walker_loss_tst'].append(walker_loss_tst.item())
            val_log['walker_loss_sts'].append(walker_loss_sts.item())
            val_log['visit_loss_ts'].append(visit_loss_ts.item())
            val_log['visit_loss_st'].append(visit_loss_st.item())
            val_log['metric_loss_st'].append(metric_loss_st.item())
            val_log['metric_loss_tt'].append(metric_loss_tt.item())

        return val_log

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
        print("[val]   metric_loss_st: %f, metric_loss_tt: %f\n" % (
            np.mean(val_log['metric_loss_st']),
            np.mean(val_log['metric_loss_tt'])
        ))

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
                
                # compute train_log
                walker_loss_tst = self.criterion['walker_tst'](t, s, labels)
                walker_loss_sts = self.criterion['walker_sts'](s, t, labels)
                visit_loss_ts = self.criterion['visit_ts'](t, s, labels)
                visit_loss_st = self.criterion['visit_st'](s, t, labels)

                # ML
                # TT
                embedding = t
                metric_loss_tt = self.criterion['metric_tt'](embedding, 'TT')
                # ST
                s_mask = torch.ByteTensor([[1], [0]]).repeat(t.size(0) // 2, 128).cuda()
                t_mask = torch.ByteTensor([[0], [1]]).repeat(t.size(0) // 2, 128).cuda()
                masked_s = torch.zeros(t.size(0), 128).cuda().masked_scatter(s_mask, s)
                masked_t = torch.zeros(t.size(0), 128).cuda().masked_scatter(t_mask, t)
                embedding = masked_s + masked_t
                metric_loss_st = self.criterion['metric_st'](embedding, 'ST')
                # flip t
                inverted_t = t.index_select(0, torch.LongTensor([i * 2 for i in range(t.size(0) // 2)]).cuda())
                inverted_masked_t = torch.zeros(t.size(0), 128).cuda().masked_scatter(t_mask, inverted_t)
                embedding = masked_s + inverted_masked_t
                metric_loss_st += self.criterion['metric_st'](embedding, 'ST')
                
                # accumulate loss
                train_loss = walker_loss_tst + walker_loss_sts + visit_loss_ts + visit_loss_st + configs.METRIC_MULTIPLIER * metric_loss_st + configs.METRIC_MULTIPLIER * metric_loss_tt
                train_log['forward'].append(time.time() - forward_since)

                # back prop
                self.optimizer.zero_grad()
                backward_since = time.time()
                train_loss.backward()
                clip_grad_value_(list(shape_encoder.parameters()) + list(text_encoder.parameters()), configs.CLIP_VALUE)
                self.optimizer.step()
                train_log['backward'].append(time.time() - backward_since)

                # record
                train_log['walker_loss_tst'].append(walker_loss_tst.item())
                train_log['walker_loss_sts'].append(walker_loss_sts.item())
                train_log['visit_loss_ts'].append(visit_loss_ts.item())
                train_log['visit_loss_st'].append(visit_loss_st.item())
                train_log['metric_loss_st'].append(metric_loss_st.item())
                train_log['metric_loss_tt'].append(metric_loss_tt.item())
                train_log['train_loss'].append(train_loss.item())
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

                    # save the best models
                    print("[{}] saving models...\n".format(rank))
                    torch.save(shape_encoder, "outputs/models/embeddings/shape_encoder_{}.pth".format(self.settings))
                    torch.save(text_encoder, "outputs/models/embeddings/text_encoder_{}.pth".format(self.settings))

        # done
        print("[{}] done...\n".format(rank))

        # return best['shape_encoder'], best['text_encoder']
