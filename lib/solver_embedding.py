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
from lib.configs import CONF
from torch.nn.utils import clip_grad_value_
from lib.save_embedding import extract
from lib.eval_embedding import compute_metrics
from eval_attn import evaluate
from model.encoder_attn import MultiHeadEncoder

class EmbeddingSolver():
    def __init__(self, shapenet, criterion, optimizer, settings, batch_size, is_multi_head=False):
        self.shapenet = shapenet
        self.criterion = criterion
        self.optimizer = optimizer
        self.settings = settings
        self.batch_size = batch_size
        self.is_multi_head = is_multi_head

    def _compute_loss(self, s, t, s_labels, t_labels, text_encoder, batch_size):
        # LBA
        if CONF.LBA.IS_LBA:
            # TST
            if CONF.LBA.IS_LBA_TST:
                # build uniformed target
                text_targets = t_labels.unsqueeze(0).expand(t_labels.size(0), t_labels.size(0)).eq(
                    t_labels.unsqueeze(1).expand(t_labels.size(0), t_labels.size(0))
                ).float()
                text_targets /= text_targets.sum(1)
                walker_loss_tst = self.criterion['walker'](t, s, text_targets)
                if CONF.LBA.IS_LBA_VISIT:
                    visit_loss_ts = self.criterion['visit'](t, s)
                else:
                    visit_loss_ts = torch.Tensor([0]).cuda()[0]
            else:
                walker_loss_tst = torch.Tensor([0]).cuda()[0]
                visit_loss_ts = torch.Tensor([0]).cuda()[0]
            # STS
            if CONF.LBA.IS_LBA_STS:
                # build uniformed target
                shape_targets = s_labels.unsqueeze(0).expand(s_labels.size(0), s_labels.size(0)).eq(
                    s_labels.unsqueeze(1).expand(s_labels.size(0), s_labels.size(0))
                ).float()
                shape_targets /= shape_targets.sum(1)
                walker_loss_sts = self.criterion['walker'](s, t, shape_targets)
                if CONF.LBA.IS_LBA_VISIT:
                    visit_loss_st = self.criterion['visit'](s, t)
                else:
                    visit_loss_st = torch.Tensor([0]).cuda()[0]
            else:
                walker_loss_sts = torch.Tensor([0]).cuda()[0]
                visit_loss_st = torch.Tensor([0]).cuda()[0]
        else:
            walker_loss_tst = torch.Tensor([0]).cuda()[0]
            visit_loss_ts = torch.Tensor([0]).cuda()[0]
            walker_loss_sts = torch.Tensor([0]).cuda()[0]
            visit_loss_st = torch.Tensor([0]).cuda()[0]

        # ML
        if CONF.ML.IS_ML:
            # TT
            if CONF.ML.IS_ML_TT:
                embedding = t
                metric_loss_tt = self.criterion['metric'](embedding)
            else:
                metric_loss_tt = torch.Tensor([0]).cuda()[0]
            if CONF.ML.IS_ML_SS:
                if text_encoder:
                    pass
                else:
                    embedding = s
                    metric_loss_tt += self.criterion['metric'](embedding)
            # ST
            if CONF.ML.IS_ML_ST:
                if text_encoder:
                    s_mask = torch.ByteTensor([[1], [0]]).repeat(batch_size // 2, 128).cuda()
                    t_mask = torch.ByteTensor([[0], [1]]).repeat(batch_size // 2, 128).cuda()
                    selected_s = s
                    selected_t = t.index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).cuda())
                    masked_s = torch.zeros(batch_size, 128).cuda().masked_scatter_(s_mask, selected_s)
                    masked_t = torch.zeros(batch_size, 128).cuda().masked_scatter_(t_mask, selected_t)
                    embedding = masked_s + masked_t
                    metric_loss_st = self.criterion['metric'](embedding)
                    # flip t
                    flipped_t = t.index_select(0, torch.LongTensor([i * 2 + 1 for i in range(batch_size // 2)]).cuda())
                    flipped_masked_t = torch.zeros(batch_size, 128).cuda().masked_scatter_(t_mask, flipped_t)
                    embedding = masked_s + flipped_masked_t
                    metric_loss_st += self.criterion['metric'](embedding)
                else:
                    s_mask = torch.ByteTensor([[1], [0]]).repeat(batch_size // 2, 128).cuda()
                    t_mask = torch.ByteTensor([[0], [1]]).repeat(batch_size // 2, 128).cuda()
                    # upper_s to upper_t
                    selected_s = s.index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).cuda())
                    selected_t = t.index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).cuda())
                    masked_s = torch.zeros(batch_size, 128).cuda().masked_scatter_(s_mask, selected_s)
                    masked_t = torch.zeros(batch_size, 128).cuda().masked_scatter_(t_mask, selected_t)
                    embedding = masked_s + masked_t
                    metric_loss_st = self.criterion['metric'](embedding)
                    # lower_s to upper_t
                    selected_s = s.index_select(0, torch.LongTensor([i * 2 + 1 for i in range(batch_size // 2)]).cuda())
                    selected_t = t.index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).cuda())
                    masked_s = torch.zeros(batch_size, 128).cuda().masked_scatter_(s_mask, selected_s)
                    masked_t = torch.zeros(batch_size, 128).cuda().masked_scatter_(t_mask, selected_t)
                    embedding = masked_s + masked_t
                    metric_loss_st += self.criterion['metric'](embedding)
                    # upper_s to lower_t
                    selected_s = s.index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).cuda())
                    selected_t = t.index_select(0, torch.LongTensor([i * 2 + 1 for i in range(batch_size // 2)]).cuda())
                    masked_s = torch.zeros(batch_size, 128).cuda().masked_scatter_(s_mask, selected_s)
                    masked_t = torch.zeros(batch_size, 128).cuda().masked_scatter_(t_mask, selected_t)
                    embedding = masked_s + masked_t
                    metric_loss_st += self.criterion['metric'](embedding)
                    # lower_s to lower_t
                    selected_s = s.index_select(0, torch.LongTensor([i * 2 + 1 for i in range(batch_size // 2)]).cuda())
                    selected_t = t.index_select(0, torch.LongTensor([i * 2 + 1 for i in range(batch_size // 2)]).cuda())
                    masked_s = torch.zeros(batch_size, 128).cuda().masked_scatter_(s_mask, selected_s)
                    masked_t = torch.zeros(batch_size, 128).cuda().masked_scatter_(t_mask, selected_t)
                    embedding = masked_s + masked_t
                    metric_loss_st += self.criterion['metric'](embedding)
            else:
                metric_loss_st = torch.Tensor([0]).cuda()[0]
        else:
            metric_loss_tt = torch.Tensor([0]).cuda()[0]
            metric_loss_st = torch.Tensor([0]).cuda()[0]
        
        # add norm penalty
        if CONF.NORM.IS_NORM_PENALTY:
            shape_norm_penalty = self._norm_penalty(s)
            text_norm_penalty = self._norm_penalty(t)
        else:
            shape_norm_penalty = torch.Tensor([0]).cuda()[0]
            text_norm_penalty = torch.Tensor([0]).cuda()[0]

        # accumulate loss
        loss = walker_loss_tst + walker_loss_sts + visit_loss_ts + visit_loss_st
        loss += CONF.ML.METRIC_MULTIPLIER * (metric_loss_st + metric_loss_tt)
        loss += (CONF.NORM.SHAPE_NORM_MULTIPLIER * shape_norm_penalty + CONF.NORM.TEXT_NORM_MULTIPLIER * text_norm_penalty)

        losses = {
            'total_loss': loss,
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
    
    def _merge_loss(self, ext_loss, attn_loss):
        losses = {}
        for key in ext_loss.keys():
            losses[key] = ext_loss[key] + CONF.MH.ATTN_MULTIPLIER * attn_loss[key]
        
        return losses

    def forward(self, shape_encoder, text_encoder, shapes, texts, labels):
        # load
        batch_size = shapes.size(0)
        texts = texts.cuda()
        text_labels = labels.cuda()
        if text_encoder:
            shapes = shapes.cuda().index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).cuda())
            shape_labels = labels.cuda().index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).cuda())
        else:
            shapes = shapes.cuda()
            shape_labels = labels.cuda()
        
        # forward pass
        if text_encoder:
            s = shape_encoder(shapes)
            t = text_encoder(texts)
            losses = self._compute_loss(s, t, shape_labels, text_labels, text_encoder, batch_size)
        else:
            if self.is_multi_head:
                s, t, s_attn, t_attn, _, _ = shape_encoder(shapes, texts)
                ext_loss = self._compute_loss(s, t, shape_labels, text_labels, text_encoder, batch_size)
                attn_loss = self._compute_loss(s_attn, t_attn, shape_labels, text_labels, text_encoder, batch_size)
                losses = self._merge_loss(ext_loss, attn_loss)
            else:
                s, t, _, _ = shape_encoder(shapes, texts)
                losses = self._compute_loss(s, t, shape_labels, text_labels, text_encoder, batch_size)
        

        return losses

    def validate(self, shape_encoder, text_encoder, dataloader, val_log):
        print("validating...\n")
        for _, (_, shapes, texts, _, labels, _) in enumerate(dataloader['val']):
            start = time.time()
            # forward pass
            # eval mode
            shape_encoder.eval()
            if text_encoder:
                text_encoder.eval()
            losses = self.forward(shape_encoder, text_encoder, shapes, texts, labels)

            # record
            val_log['total_loss'].append(losses['total_loss'].item())
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

    def evaluate(self, shape_encoder, text_encoder, eval_dataloader, iter_time_list):
        # # extract embedding
        # print("extracting...\n")
        # # eval mode
        # shape_encoder.eval()
        # if text_encoder:
        #     text_encoder.eval()
        # embedding = extract(shape_encoder, text_encoder, eval_dataloader, None, None)

        # evaluate
        mean_exe_s = np.mean(iter_time_list)
        eta_s = mean_exe_s * len(eval_dataloader['shape']) * len(eval_dataloader['text']) * 2
        eta_m = math.floor(eta_s / 60)
        print("evaluating, ETA: {}m {}s\n".format(int(eta_m), int(eta_s - eta_m * 60)))
        metrics_t2s, metrics_s2t = evaluate(
            shape_encoder, 
            text_encoder, 
            eval_dataloader['shape'], 
            eval_dataloader['text'], 
            self.batch_size,
            0, 
            self.is_multi_head
        )

        return metrics_t2s, metrics_s2t

    def train(self, shape_encoder, text_encoder, best, dataloader, eval_dataloader, epoch, verbose):
        '''
        note:
            text_encoder = None -> attention is applied
        '''
        total_iter = len(dataloader['train']) * epoch
        iter_count = 0
        scheduler = StepLR(self.optimizer, step_size=CONF.TRAIN.REDUCE_STEP, gamma=CONF.TRAIN.REDUCE_FACTOR)
        log = {
            'train': {
                'total_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_st': [],
                'metric_loss_tt': [],
                'shape_norm_penalty': [],
                'text_norm_penalty': []
            },
            'val': {
                'total_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_st': [],
                'metric_loss_tt': [],
                'shape_norm_penalty': [],
                'text_norm_penalty': []
            },
            'eval': {
                'eval_time': []
            }
        }
        for epoch_id in range(epoch):
            print("epoch [{}/{}] starting...\n".format(epoch_id+1, epoch))
            scheduler.step()
            train_log = {
                'forward': [],
                'backward': [],
                'iter_time': [],
                'total_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_st': [],
                'metric_loss_tt': [],
                'shape_norm_penalty': [],
                'text_norm_penalty': [],
                'fetch_time': []
            }
            val_log = {
                'iter_time': [],
                'total_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_st': [],
                'metric_loss_tt': [],
                'shape_norm_penalty': [],
                'text_norm_penalty': []
                
            }
            for _, (_, shapes, texts, _, labels, fetch_time) in enumerate(dataloader['train']):
                start = time.time()
                # forward pass
                forward_since = time.time()
                # train mode
                shape_encoder.train()
                if text_encoder:
                    text_encoder.train()
                losses = self.forward(shape_encoder, text_encoder, shapes, texts, labels)
                train_log['forward'].append(time.time() - forward_since)
                # record
                train_log['total_loss'].append(losses['total_loss'].item())
                train_log['walker_loss_tst'].append(losses['walker_loss_tst'].item())
                train_log['walker_loss_sts'].append(losses['walker_loss_sts'].item())
                train_log['visit_loss_ts'].append(losses['visit_loss_ts'].item())
                train_log['visit_loss_st'].append(losses['visit_loss_st'].item())
                train_log['metric_loss_st'].append(losses['metric_loss_st'].item())
                train_log['metric_loss_tt'].append(losses['metric_loss_tt'].item())
                train_log['shape_norm_penalty'].append(losses['shape_norm_penalty'].item())
                train_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())
                train_log['fetch_time'].append(fetch_time)

                # back prop
                self.optimizer.zero_grad()
                backward_since = time.time()
                losses['total_loss'].backward()
                if text_encoder:
                    clip_grad_value_(list(shape_encoder.parameters()) + list(text_encoder.parameters()), CONF.TRAIN.CLIP_VALUE)
                else:
                    clip_grad_value_(shape_encoder.parameters(), CONF.TRAIN.CLIP_VALUE)
                self.optimizer.step()
                train_log['backward'].append(time.time() - backward_since)

                # record
                train_log['iter_time'].append(time.time() - start + fetch_time)
                iter_count += 1

                # report
                if iter_count % verbose == 0:
                    val_est = np.mean(train_log['iter_time'] + val_log['iter_time']) * len(dataloader['val']) * (epoch - epoch_id - 1)
                    eval_est = 0
                    self._iter_report(train_log, iter_count, total_iter, val_est, eval_est)

                # # evaluate
                # if iter_count % CONF.TRAIN.EVAL_FREQ == 0:
                    # self.evaluate(shape_encoder, text_encoder, eval_dataloader)

            # validate
            val_log = self.validate(shape_encoder, text_encoder, dataloader, val_log)

            # # evaluate
            # metrics_t2s, metrics_s2t = self.evaluate(shape_encoder, text_encoder, eval_dataloader, train_log['iter_time'] + val_log['iter_time'])
            # scores = metrics_t2s.recall_rate[0] + metrics_t2s.recall_rate[4] + metrics_t2s.ndcg[4]
            # scores += metrics_s2t.recall_rate[0] + metrics_s2t.recall_rate[4] + metrics_s2t.ndcg[4]
            scores = 0
            
            # epoch report
            self._epoch_report(train_log, val_log, epoch_id, epoch)
            
            # # best
            # if scores > best['scores']:
            #     # report best
            #     print("best scores achieved: {}".format(scores))
            #     print("current train_loss: {}".format(np.mean(train_log['total_loss'])))
            #     print("current val_loss: {}".format(np.mean(val_log['total_loss'])))
            #     best['epoch'] = epoch_id + 1
            #     best['scores'] = scores
            #     best['total_loss'] = float(np.mean(val_log['total_loss']))
            #     best['walker_loss_tst'] = float(np.mean(val_log['walker_loss_tst']))
            #     best['walker_loss_sts'] = float(np.mean(val_log['walker_loss_sts']))
            #     best['visit_loss_ts'] = float(np.mean(val_log['visit_loss_ts']))
            #     best['visit_loss_st'] = float(np.mean(val_log['visit_loss_st']))
            #     best['metric_loss_st'] = float(np.mean(val_log['metric_loss_st']))
            #     best['metric_loss_tt'] = float(np.mean(val_log['metric_loss_tt']))
            #     best['shape_norm_penalty'] = float(np.mean(val_log['shape_norm_penalty']))
            #     best['text_norm_penalty'] = float(np.mean(val_log['text_norm_penalty']))

            #     # save the best models
            #     print("saving models...\n")
            #     if not os.path.exists(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models")):
            #         os.mkdir(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models"))
            #     if text_encoder:
            #         torch.save(shape_encoder.state_dict(), os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models", "shape_encoder.pth"))
            #         torch.save(text_encoder.state_dict(), os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models", "text_encoder.pth"))
            #     else:
            #         torch.save(shape_encoder.state_dict(), os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models", "encoder.pth"))
            
            # best
            if np.mean(val_log['total_loss']) < best['total_loss']:
                # report best
                print("best val_loss achieved: {}".format(np.mean(val_log['total_loss'])))
                # print("current scores: {}".format(scores))
                print("current train_loss: {}".format(np.mean(train_log['total_loss'])))
                best['epoch'] = epoch_id + 1
                best['scores'] = scores
                best['total_loss'] = float(np.mean(val_log['total_loss']))
                best['walker_loss_tst'] = float(np.mean(val_log['walker_loss_tst']))
                best['walker_loss_sts'] = float(np.mean(val_log['walker_loss_sts']))
                best['visit_loss_ts'] = float(np.mean(val_log['visit_loss_ts']))
                best['visit_loss_st'] = float(np.mean(val_log['visit_loss_st']))
                best['metric_loss_st'] = float(np.mean(val_log['metric_loss_st']))
                best['metric_loss_tt'] = float(np.mean(val_log['metric_loss_tt']))
                best['shape_norm_penalty'] = float(np.mean(val_log['shape_norm_penalty']))
                best['text_norm_penalty'] = float(np.mean(val_log['text_norm_penalty']))

                # save the best models
                print("saving models...\n")
                if not os.path.exists(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models")):
                    os.mkdir(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models"))
                if text_encoder:
                    torch.save(shape_encoder.state_dict(), os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models", "shape_encoder.pth"))
                    torch.save(text_encoder.state_dict(), os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models", "text_encoder.pth"))
                else:
                    torch.save(shape_encoder.state_dict(), os.path.join(CONF.PATH.OUTPUT_EMBEDDING, self.settings, "models", "encoder.pth"))

            # epoch log
            log['train']['total_loss'].append(np.mean(train_log['total_loss']))
            log['train']['walker_loss_tst'].append(np.mean(train_log['walker_loss_tst']))
            log['train']['walker_loss_sts'].append(np.mean(train_log['walker_loss_sts']))
            log['train']['visit_loss_ts'].append(np.mean(train_log['visit_loss_ts']))
            log['train']['visit_loss_st'].append(np.mean(train_log['visit_loss_st']))
            log['train']['metric_loss_st'].append(np.mean(train_log['metric_loss_st']))
            log['train']['metric_loss_tt'].append(np.mean(train_log['metric_loss_tt']))
            log['train']['shape_norm_penalty'].append(np.mean(train_log['shape_norm_penalty']))
            log['train']['text_norm_penalty'].append(np.mean(train_log['text_norm_penalty']))
            log['val']['total_loss'].append(np.mean(val_log['total_loss']))
            log['val']['walker_loss_tst'].append(np.mean(val_log['walker_loss_tst']))
            log['val']['walker_loss_sts'].append(np.mean(val_log['walker_loss_sts']))
            log['val']['visit_loss_ts'].append(np.mean(val_log['visit_loss_ts']))
            log['val']['visit_loss_st'].append(np.mean(val_log['visit_loss_st']))
            log['val']['metric_loss_st'].append(np.mean(val_log['metric_loss_st']))
            log['val']['metric_loss_tt'].append(np.mean(val_log['metric_loss_tt']))
            log['val']['shape_norm_penalty'].append(np.mean(val_log['shape_norm_penalty']))
            log['val']['text_norm_penalty'].append(np.mean(val_log['text_norm_penalty']))

        # done
        print("done...\n")

        # end eval
        self.evaluate(shape_encoder, text_encoder, eval_dataloader, train_log['iter_time'] + val_log['iter_time'])

        return best, log

    def _norm_penalty(self, embedding):
        '''
        added penalty to the loss if embedding's loss exceeds the threshold
        '''
        norm = torch.norm(embedding, p=2, dim=1)
        penalty = torch.max(torch.zeros(norm.size()).cuda(), norm - CONF.NORM.MAX_NORM).mean()

        return penalty

    def _iter_report(self, log, iter_count, total_iter, val_est, eval_est):
        # compute ETA
        exetime_s = np.mean(log['iter_time'])
        eta_s = exetime_s * (total_iter - iter_count) + val_est + eval_est
        eta_m = math.floor(eta_s / 60)
        
        # show report
        print("----------------------iter: [{}/{}]----------------------".format(iter_count, total_iter))
        print("[loss] total_loss: %f" % (
            np.mean(log['total_loss'])
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
        print("[info] forward_per_iter: %fs\n[info] backward_per_iter: %fs\n[info] data_fetch: %fs" % ( 
            np.mean(log['forward']),
            np.mean(log['backward']),
            np.mean(log['fetch_time'])
        ))
        print("[info] time_per_iter: %fs\n[info] ETA: %dm %ds\n" % ( 
            np.mean(log['iter_time']),
            eta_m,
            eta_s - eta_m * 60
        ))

    def _epoch_report(self, train_log, val_log, epoch_id, epoch):
        # show report
        print("epoch [{}/{}] done...".format(epoch_id+1, epoch))
        print("------------------------summary------------------------")
        print("[train] total_loss: %f" % (
            np.mean(train_log['total_loss'])
        ))
        print("[val]   total_loss: %f" % (
            np.mean(val_log['total_loss'])
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
