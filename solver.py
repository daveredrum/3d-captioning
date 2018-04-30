import torch
import time
import math
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from nltk.translate.bleu_score import sentence_bleu


# only for encoder offline training 
class EncoderSolver():
    def __init__(self, optimizer, criterion, cuda_flag=True):
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda_flag = cuda_flag
        self.log = {}

    def train(self, model, dataloader, epoch, verbose):
        for epoch_id in range(epoch):
            log = {
                'train_loss': [],
                'train_acc': [],
                'valid_loss': [],
                'valid_acc': []    
            }
            start = time.time()
            for phase in ['train', 'valid']:
                if phase == "valid":
                    model.eval()
                else:
                    model.train()
                for image, label in dataloader[phase]:
                    if self.cuda_flag:
                        inputs = Variable(image).cuda()
                        targets = Variable(label).cuda()
                    else:
                        inputs = Variable(image)
                        targets = Variable(label)
                    # feed the model
                    outputs = model(inputs)
                    # compute loss
                    self.optimizer.zero_grad()
                    loss = self.criterion(outputs, targets)
                    # compute accuracy
                    _, preds = torch.max(outputs, 1)
                    if self.cuda_flag:
                        acc = accuracy_score(preds.data.cpu().numpy(), targets.data.cpu().numpy())
                    else:
                        acc = accuracy_score(preds.data.numpy(), targets.data.numpy())
                    # record
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                        log['train_loss'].append(loss.data[0])
                        log['train_acc'].append(acc)
                        # print("train", log['train_loss'][-1], log['train_acc'][-1])
                    else:
                        log['valid_loss'].append(loss.data[0])
                        log['valid_acc'].append(acc)
                        # print("valid", log['valid_loss'][-1], log['valid_acc'][-1])
            # show report
            if epoch_id % verbose == (verbose - 1):
                exetime_s = time.time() - start
                eta_s = exetime_s * (epoch - (epoch_id + 1))
                eta_m = math.floor(eta_s / 60)
                print("[epoch %d]: train_loss: %f, train_acc: %f, valid_loss: %f, valid_acc: %f, ETA: %dm %ds" % (
                    epoch_id + 1, 
                    np.mean(log['train_loss']), 
                    np.mean(log['train_acc']),
                    np.mean(log['valid_loss']), 
                    np.mean(log['valid_acc']),
                    eta_m,
                    eta_s - eta_m * 60

                ))
            # save log
            log['train_loss'] = np.mean(log['train_loss'])
            log['train_acc'] = np.mean(log['train_acc'])
            log['valid_loss'] = np.mean(log['valid_loss'])
            log['valid_acc'] = np.mean(log['valid_acc'])
            self.log[epoch_id] = log
            # save model
            torch.save(model, "data/encoder.pth")

# only for decoder offline training
# extracted visual contexts required
class DecoderSolver():
    def __init__(self, optimizer, criterion, cuda_flag=True):
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda_flag = cuda_flag
        self.log = {}

    def train(self, model, dataloader, epoch, verbose):
        # training_pairs is a list of (visual_context, caption)
        # visual_context and caption are both numpy arrays
        for epoch_id in range(epoch):
            log = {
                'train_loss': [],
                'valid_loss': []   
            }
            start = time.time()
            for phase in ["train", "valid"]:
                for visuals, captions, cap_lengths in dataloader[phase]:
                    caption_inputs = torch.cat([item.view(1, -1) for item in captions]).transpose(1, 0)[:, :cap_lengths[0]-1]
                    caption_targets = torch.cat([item.view(1, -1) for item in captions]).transpose(1, 0)[:, :cap_lengths[0]]
                    if self.cuda_flag:
                        visual_inputs = Variable(visuals).cuda()
                        caption_inputs = Variable(caption_inputs).cuda()
                        caption_targets = Variable(caption_targets).cuda()
                        caption_targets = pack_padded_sequence(caption_targets, cap_lengths, batch_first=True)[0]
                        cap_lengths = Variable(cap_lengths).cuda()
                    else:
                        visual_inputs = Variable(visuals)
                        caption_inputs = Variable(caption_inputs)
                        caption_targets = Variable(caption_targets)
                        caption_targets = pack_padded_sequence(caption_targets, cap_lengths, batch_first=True)[0]
                        cap_lengths = Variable(cap_lengths)
                    outputs, _ = model(visual_inputs, caption_inputs, cap_lengths)
                    loss = self.criterion(outputs, caption_targets)
                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        log['train_loss'].append(loss.data[0])
                    else:
                        log['valid_loss'].append(loss.data[0])
            # show report
            if epoch_id % verbose == (verbose - 1):
                exetime_s = time.time() - start
                eta_s = exetime_s * (epoch - (epoch_id + 1))
                eta_m = math.floor(eta_s / 60)
                print("[epoch %d/%d] train_loss: %f, valid_loss: %f, ETA: %dm %ds" % (
                    epoch_id + 1,
                    epoch, 
                    np.mean(log['train_loss']), 
                    np.mean(log['valid_loss']),
                    eta_m,
                    eta_s - eta_m * 60
                    ))
            # save log
            log['train_loss'] = np.mean(log['train_loss'])
            log['valid_loss'] = np.mean(log['valid_loss'])
            self.log[epoch_id] = log
            # save model
            torch.save(model, "data/decoder.pth")

class EncoderDecoderSolver():
    def __init__(self, optimizer, criterion, model_type, cuda_flag=True):
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_type = model_type
        self.cuda_flag = cuda_flag
        self.log = {}
    
    def _unpack_outputs(self, sequence, pack_info):
        # unpack the sequence
        
        return pad_packed_sequence(PackedSequence(sequence.max(1)[1], pack_info))[0].transpose(1, 0)

    def _unpad_outputs(self, unpacked_sequence, cap_lengths):
        # unpad the sequence and return sequence lists
        
        return [unpacked_sequence[i][:cap_lengths[i]].tolist() for i in range(cap_lengths.size(0))]

    def _calculate_blue(self, corpus, dictionary, model_ids, sequences):
        blue = []
        for model_id, sequence in zip(model_ids, sequences):
            sentence = []
            for idx in sequence:
                try:
                    sentence.append(dictionary[idx])
                except Exception:
                    pass
            blue.append(
                sentence_bleu(
                    corpus[model_id],
                    sentence
                )
            )
        
        return np.mean(blue)

    def train(self, encoder, decoder, dataloader, corpus, dictionary, epoch, verbose):
        for epoch_id in range(epoch):
            log = {
                'train_loss': [],
                'train_blue': [],
                'valid_loss': [],
                'valid_blue': [],
                'forward': [],
                'backward': []
            }
            start = time.time()
            for phase in ["train", "valid"]:
                if phase == "valid":
                    encoder.eval()
                else:
                    encoder.train()
                for model_ids, visuals, captions, cap_lengths in dataloader[phase]:
                    # visuals must be tensor
                    caption_inputs = torch.cat([item.view(1, -1) for item in captions]).transpose(1, 0)[:, :cap_lengths[0]-1]
                    caption_targets = torch.cat([item.view(1, -1) for item in captions]).transpose(1, 0)[:, :cap_lengths[0]]
                    # target_ref = [item[0][:item[1]] for item in zip(caption_targets.tolist(), cap_lengths.tolist())]
                    
                    if self.cuda_flag:
                        visual_inputs = Variable(visuals).cuda()
                        caption_inputs = Variable(caption_inputs).cuda()
                        caption_targets = Variable(caption_targets).cuda()
                        caption_targets, pack_info = pack_padded_sequence(caption_targets, cap_lengths, batch_first=True)
                        cap_lengths = Variable(cap_lengths).cuda()
                    else:
                        visual_inputs = Variable(visuals)
                        caption_inputs = Variable(caption_inputs)
                        caption_targets = Variable(caption_targets)
                        caption_targets, pack_info = pack_padded_sequence(caption_targets, cap_lengths, batch_first=True)
                        cap_lengths = Variable(cap_lengths)
                    
                    forward_since = time.time()
                    visual_contexts = encoder.extract(visual_inputs)
                    outputs, _ = decoder(visual_contexts, caption_inputs, cap_lengths)
                    log['forward'].append(time.time() - forward_since)
                    loss = self.criterion(outputs, caption_targets)
                    
                    # unpack outputs
                    outputs_unpack = self._unpack_outputs(outputs, pack_info)
                    # unpadd outputs
                    outputs_unpad = self._unpad_outputs(outputs_unpack, cap_lengths)
                    # calculate BLEU score
                    blue = self._calculate_blue(corpus, dictionary, model_ids, outputs_unpad)

                    if phase == "train":
                        encoder.zero_grad()
                        decoder.zero_grad()
                        backward_since = time.time()
                        loss.backward()
                        log['backward'].append(time.time() - backward_since)
                        self.optimizer.step()
                        log['train_loss'].append(loss.data[0])
                        log['train_blue'].append(np.mean(blue))
                    else:
                        log['valid_loss'].append(loss.data[0])
                        log['valid_blue'].append(np.mean(blue))
            # show report
            if epoch_id % verbose == (verbose - 1):
                exetime_s = time.time() - start
                eta_s = exetime_s * (epoch - (epoch_id + 1))
                eta_m = math.floor(eta_s / 60)
                print("-----------------epoch %d/%d-----------------\n[Info] train_loss: %f, train_blue: %f\n[Info] valid_loss: %f, valid_blue: %f\n[Info] forward: %fs, backward: %fs\n[Info] ETA: %dm %ds\n" % (
                    epoch_id + 1,
                    epoch, 
                    np.mean(log['train_loss']), 
                    np.mean(log['train_blue']),
                    np.mean(log['valid_loss']),
                    np.mean(log['valid_blue']),
                    np.mean(log['forward']), 
                    np.mean(log['backward']),
                    eta_m,
                    eta_s - eta_m * 60
                    ))
            # save log
            log['train_loss'] = np.mean(log['train_loss'])
            log['valid_loss'] = np.mean(log['valid_loss'])
            self.log[epoch_id] = log
            # save model
            torch.save(encoder, "data/encoder_checkpoint_%s.pth" % self.model_type)
            torch.save(decoder, "data/decoder_checkpoint_%s.pth" % self.model_type)

                