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


class RoundTripLoss(nn.Module):
    def __init__(self, weight=1.):
        super(RoundTripLoss, self).__init__()
        self.weight = weight
    
    def forward(self, a, b, labels):
        '''
        params: 
            a: 2D embedding tensor, either text embeddings or shape embeddings
            b: 2D embedding tensor, either text embeddings or shape embeddings

        returns:
            loss: a positive value of cross entropy loss 
        '''
        
        # build target 
        targets = (labels.unsqueeze(1).matmul(labels.unsqueeze(0)) == 1).float()
        targets /= targets.sum(1)
        # similarity
        sim = a.matmul(b.transpose(1, 0).contiguous())
        # walk
        a2b = F.softmax(sim, dim=1)
        b2a = F.softmax(sim.transpose(1, 0), dim=1)
        # build inputs
        inputs = a2b.matmul(b2a).log()

        return -self.weight * targets.mul(1e-8 + inputs).sum(1).mean()


class AssociationLoss(nn.Module):
    def __init__(self, weight=1.):
        super(AssociationLoss, self).__init__()
        self.weight = weight
    
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
        inputs = a2b.mean(0, keepdim=True).log()
        # build targets
        if inputs.is_cuda:
            targets = torch.FloatTensor(1, inputs.size(1)).fill_(1. / inputs.size(1)).cuda()
        else:
            targets = torch.FloatTensor(1, inputs.size(1)).fill_(1. / inputs.size(1))

        return -self.weight * targets.mul(1e-8 + inputs).sum(1).mean()


class MetricLoss(nn.Module):
    def __init__(self, margin=1.):
        super(MetricLoss, self).__init__()
        self.margin = margin
    
    def _build_pairs(self, labels):
        '''
        params:
            labels: 1D tensor, labels of samples
        
        return:
            pos: indices of positive set
            neg: indices of negative set
        '''
        pos, neg = [], []
        for pair in itertools.permutations(range(labels.size(0)), 2):
            if labels[pair[0]] == labels[pair[1]]:
                pos.append(pair)
            else:
                neg.append(pair)

        return pos, neg

    # def forward(self, a, b, labels):
    #     '''
    #     param:
    #         a: 2D embedding tensor, either text embeddings or shape embeddings
    #         b: 2D embedding tensor, either text embeddings or shape embeddings
    #         labels: 1D tensor, labels of samples
        
    #     return:
    #         metric_loss: see https://arxiv.org/pdf/1803.08495.pdf Sec.4.3
    #     '''
    #     pos, neg = self._build_pairs(labels)
    #     global_comp = []
    #     for pos_pair in pos:
    #         neg_i = [item for item in neg if item[0] == pos_pair[0]]
    #         neg_j = [item for item in neg if item[0] == pos_pair[1]]
    #         if a.is_cuda:
    #             V_i = torch.sum(torch.FloatTensor([(self.margin + a[item[0]].dot(b[item[1]])).exp() for item in neg_i])).cuda()
    #             V_j = torch.sum(torch.FloatTensor([(self.margin + a[item[0]].dot(b[item[1]])).exp() for item in neg_j])).cuda()
    #             max_ij = (V_i + V_j).log() - a[pos_pair[0]].dot(b[pos_pair[1]])
    #             # print("sim: {}, dis: {}, max: {}".format(a[pos_pair[0]].dot(b[pos_pair[1]]), (V_i + V_j).log(), max_ij))
    #             max_ij = torch.max(max_ij, torch.zeros(max_ij.size()).cuda())
    #         else:
    #             V_i = torch.sum(torch.FloatTensor([(self.margin + a[item[0]].dot(b[item[1]])).exp() for item in neg_i]))
    #             V_j = torch.sum(torch.FloatTensor([(self.margin + a[item[0]].dot(b[item[1]])).exp() for item in neg_j]))
    #             max_ij = (V_i + V_j).log() - a[pos_pair[0]].dot(b[pos_pair[1]])  
    #             max_ij = torch.max(max_ij, torch.zeros(max_ij.size()))
                
    #         local_comp = max_ij.pow(2)
    #         global_comp.append(local_comp.unsqueeze(0))
    #     outputs = torch.cat(global_comp).sum().div(2 * len(pos))
    #     if a.is_cuda:
    #         outputs.cuda()
        
    #     return outputs

    def forward(self, a, b, labels):
        '''
        param:
            a: 2D embedding tensor, either text embeddings or shape embeddings
            b: 2D embedding tensor, either text embeddings or shape embeddings
            labels: 1D tensor, labels of samples
        
        return:
            metric_loss: see https://arxiv.org/pdf/1511.06452.pdf Sec.4
        '''
        pos, neg = self._build_pairs(labels)
        Xa = a.clone().unsqueeze(0)
        Xb = b.clone().unsqueeze(1)
        Dsq = torch.sum(torch.pow(Xa - Xb, 2), dim=2)
        D = torch.sqrt(Dsq)
        Dexpm = torch.exp(self.margin - D)
        global_comp = [0.] * len(pos)
        for pos_id, pos_pair in enumerate(pos):
            # neg_i = [item for item in neg if item[0] == pos_pair[0]]
            # neg_j = [item for item in neg if item[0] == pos_pair[1]]
            # neg_ik = np.sum([Dexpm[item].item() for item in neg_i])
            # neg_jl = np.sum([Dexpm[item].item() for item in neg_j])
            # Dissim = neg_ik + neg_jl
            # if a.is_cuda:
            #     J_ij = torch.log(torch.FloatTensor([Dissim])[0]).cuda() + D[pos_pair]
            #     max_ij = torch.max(J_ij, torch.zeros(J_ij.size()).cuda()).pow(2)
            # else:
            #     J_ij = torch.log(torch.FloatTensor([Dissim])[0]) + D[pos_pair]
            #     max_ij = torch.max(J_ij, torch.zeros(J_ij.size())).pow(2)

            neg_i = [item[0] * D.size(1) + item[1] for item in neg if item[0] == pos_pair[0]]
            neg_j = [item[0] * D.size(1) + item[1] for item in neg if item[0] == pos_pair[1]]
            if a.is_cuda:
                neg_ik = Dexpm.take(torch.LongTensor(neg_i).cuda()).sum()
                neg_jl = Dexpm.take(torch.LongTensor(neg_j).cuda()).sum()
                Dissim = neg_ik + neg_jl
                J_ij = torch.log(Dissim).cuda() + D[pos_pair]
                max_ij = torch.max(J_ij, torch.zeros(J_ij.size()).cuda()).pow(2)
            else:
                neg_ik = Dexpm.take(torch.LongTensor(neg_i)).sum()
                neg_jl = Dexpm.take(torch.LongTensor(neg_j)).sum()
                Dissim = neg_ik + neg_jl
                J_ij = torch.log(Dissim) + D[pos_pair]
                max_ij = torch.max(J_ij, torch.zeros(J_ij.size())).pow(2)
            
            global_comp[pos_id] = max_ij.unsqueeze(0)
        
        outputs = torch.cat(global_comp).sum().div(2 * len(pos))
        if a.is_cuda:
            outputs.cuda()

        return outputs
