'''
############################################################

# implementation of https://arxiv.org/pdf/1803.08495.pdf

# by Dave Zhenyu Chen

############################################################
'''

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.configs as configs


class RoundTripLoss(nn.Module):
    def __init__(self, weight=1.):
        super(RoundTripLoss, self).__init__()
        self.weight = weight
        # self.klloss = nn.KLDivLoss()
    
    def forward(self, a, b, labels):
        '''
        params: 
            a: 2D embedding tensor, either text embeddings or shape embeddings
            b: 2D embedding tensor, either text embeddings or shape embeddings

        returns:
            loss: a positive value of cross entropy loss 
        '''
        
        # build target
        targets = labels.unsqueeze(0).expand(labels.size(0), labels.size(0)).eq(
            labels.unsqueeze(1).expand(labels.size(0), labels.size(0))
        ).float() / configs.N_CAPTION_PER_MODEL

        targets /= targets.sum(1)
        # similarity
        sim = a.matmul(b.transpose(1, 0).contiguous())
        # walk
        a2b = F.softmax(sim, dim=1)
        b2a = F.softmax(sim.transpose(1, 0), dim=1)
        # build inputs
        inputs = a2b.matmul(b2a)

        return -self.weight * targets.mul(torch.log(1e-8 + inputs)).sum(1).mean()
        # return self.weight * self.klloss(inputs.log(), targets)


class AssociationLoss(nn.Module):
    def __init__(self, weight=1.):
        super(AssociationLoss, self).__init__()
        self.weight = weight
        # self.klloss = nn.KLDivLoss()
    
    def forward(self, a, b, labels):
        '''
        params: 
            a: 2D embedding tensor, either text embeddings or shape embeddings
            b: 2D embedding tensor, either text embeddings or shape embeddings

        returns:
            loss: a positive value of cross entropy loss 
        '''

        # similarity
        sim = a.matmul(b.transpose(1, 0).contiguous())
        # visit
        a2b = F.softmax(sim, dim=1)
        # build inputs
        inputs = a2b.mean(0, keepdim=True)
        # build targets
        targets = torch.FloatTensor(inputs.size()).fill_(1. / inputs.size(1)).cuda()

        return -self.weight * targets.mul(torch.log(1e-8 + inputs)).sum(1).mean()
        # return self.weight * self.klloss(inputs.log(), targets)


class InstanceMetricLoss(nn.Module):
    def __init__(self, margin=1.):
        super(InstanceMetricLoss, self).__init__()
        self.margin = margin
    
    def _get_distance(self, inputs):
        # get distance
        if configs.COSINE_DISTANCE:
            D = inputs.matmul(inputs.transpose(1, 0))
            if configs.INVERTED_LOSS:
                D /= 128.
            else:
                D = 1. - D
        else:
            Xa = inputs.unsqueeze(0)
            Xb = inputs.unsqueeze(1)
            Dsq = torch.sum(torch.pow(Xa - Xb, 2), dim=2)
            D = torch.sqrt(Dsq)

        # get exponential distance
        if configs.INVERTED_LOSS:
            Dexpm = torch.exp(self.margin + D)
        else:
            Dexpm = torch.exp(self.margin - D)

        return D, Dexpm
        

    def forward(self, inputs, mode):
        '''
        instance-level metric learning loss, assuming all inputs are from different catagories
        labels are not needed

        param:
            a: 2D embedding tensor, either text embeddings or shape embeddings
            b: 2D embedding tensor, either text embeddings or shape embeddings
        
        return:
            instance-level metric_loss: see https://arxiv.org/pdf/1511.06452.pdf Sec.4.2
        '''

        batch_size = inputs.size(0)
        
        # get distance
        D, Dexpm = self._get_distance(inputs)

        # compute pair-wise loss
        global_comp = [0.] * (batch_size // 2)
        for pos_id in range(batch_size // 2):
            pos_i = pos_id * 2
            pos_j = pos_id * 2 + 1
            pos_pair = (pos_i, pos_j)
            if mode == 'TT':
                neg_i = [pos_i * batch_size + k for k in range(batch_size) if k != pos_i and k != pos_j]
                neg_j = [pos_j * batch_size + l for l in range(batch_size) if l != pos_i and l != pos_j]
            else:
                neg_i = [pos_i * batch_size + k * 2 + 1 for k in range(batch_size // 2) if k != pos_j]
                neg_j = [pos_j * batch_size + l * 2 for l in range(batch_size // 2) if l != pos_i]
            neg_ik = Dexpm.take(torch.LongTensor(neg_i).cuda()).sum()
            neg_jl = Dexpm.take(torch.LongTensor(neg_j).cuda()).sum()
            Dissim = neg_ik + neg_jl

            if configs.INVERTED_LOSS:
                J_ij = torch.log(1e-8 + Dissim).cuda() - D[pos_pair]
            else:
                J_ij = torch.log(1e-8 + Dissim).cuda() + D[pos_pair]

            max_ij = torch.max(J_ij, torch.zeros(J_ij.size()).cuda()).pow(2)            
            global_comp[pos_id] = max_ij.unsqueeze(0)
        
        # accumulate
        outputs = torch.cat(global_comp).sum().div(2 * batch_size)

        return outputs
