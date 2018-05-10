import torch
import time
import math
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from nltk.translate.bleu_score import sentence_bleu
from tensorboardX import SummaryWriter
import capeval.bleu.bleu as capbleu
import capeval.cider.cider as capcider
import capeval.meteor.meteor as capmeteor
import capeval.rouge.rouge as caprouge

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
            torch.save(model, "models/encoder.pth")

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
            torch.save(model, "models/decoder.pth")

class EncoderDecoderSolver():
    def __init__(self, optimizer, criterion, model_type, cuda_flag=True):
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_type = model_type
        self.cuda_flag = cuda_flag
        self.log = {}
    
    # # unpack the sequence
    # def _unpack_outputs(self, sequence, pack_info):
        
    #     return pad_packed_sequence(PackedSequence(sequence, pack_info))[0].transpose(1, 0)

    # # unpad the sequence by removing the zeros
    # def _unpad_outputs(self, unpacked_sequence, cap_lengths):
    #     return [unpacked_sequence[i][:cap_lengths[i]].tolist() for i in range(cap_lengths.size(0))]

    def _decode_outputs(self, sequence, pack_info, cap_lengths, dictionary):
        # unpack the sequence
        unpacked_sequence = pad_packed_sequence(PackedSequence(sequence, pack_info))[0].transpose(1, 0)
        # unpad the sequence by removing the zeros
        unpadded_sequence = [unpacked_sequence[i][:cap_lengths[i]].tolist() for i in range(cap_lengths.size(0))]
        # decode the indices
        decoded = []
        for sequence in unpadded_sequence:
            temp = []
            for idx in sequence:
                try:
                    temp.append(dictionary[idx])
                except Exception:
                    pass
            decoded.append(" ".join(temp))
        
        return decoded
    
    # calculate the bleu score with respect to the references
    def _calculate_blue(self, references, dictionary, model_ids, sequences, weights):
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
                    references[model_id],
                    sentence,
                    weights
                )
            )
        
        return np.mean(blue)

    def train(self, encoder, decoder, dataloader, references, dictionary, epoch, verbose, model_type):
        # setup tensorboard
        writer = SummaryWriter(log_dir="logs/")
        for epoch_id in range(epoch + 1):
            log = {
                'train_loss': [],
                'train_blue_1': [],
                'train_blue_2': [],
                'train_blue_3': [],
                'train_blue_4': [],
                'valid_loss': [],
                'valid_blue_1': [],
                'valid_blue_2': [],
                'valid_blue_3': [],
                'valid_blue_4': [],
                'train_cider': [],
                'valid_cider': [],
                # 'train_meteor': [],
                # 'valid_meteor': [],
                'train_rouge': [],
                'valid_rouge': [],
                'forward': [],
                'backward': [],
                'valid_time': [],
                'eval_time': [],
                'epoch_time': []
            }
            candidates = {
                'train': {},
                'valid': {}
            }
            start = time.time()
            for phase in ["train", "valid"]:
                if phase == "valid":
                    encoder.eval()
                else:
                    encoder.train()
                for model_ids, visuals, captions, cap_lengths in dataloader[phase]:
                    # visuals must be tensor
                    if model_type == "2d":
                        visuals = visuals
                    elif model_type == "3d":
                        visuals = visuals[1]
                    elif model_type == "coco":
                        visuals = visuals
                        
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
                    
                    if phase == "train":
                        # forward pass
                        forward_since = time.time()
                        visual_contexts = encoder.extract(visual_inputs)
                        # teacher forcing
                        outputs, _ = decoder(visual_contexts, caption_inputs, cap_lengths)
                        # # no teacher forcing
                        # outputs = decoder.sample(visual_contexts, cap_lengths)
                        loss = self.criterion(outputs, caption_targets)
                        log['forward'].append(time.time() - forward_since)
                        
                        # decode outputs
                        outputs = self._decode_outputs(outputs.max(1)[1], pack_info, cap_lengths, dictionary)
                        # save to candidates
                        for model_id, output in zip(model_ids, outputs):
                            if model_id not in candidates[phase].keys():
                                candidates[phase][model_id] = [output]
                            else:
                                candidates[phase][model_id].append(output)

                        # backward pass
                        # save log
                        if epoch_id != 0:
                            encoder.zero_grad()
                            decoder.zero_grad()
                            self.optimizer.zero_grad()
                            backward_since = time.time()
                            loss.backward()
                            self.optimizer.step()
                            log['backward'].append(time.time() - backward_since)
                        else:
                            log['backward'].append(0)
                        log['train_loss'].append(loss.data[0])
                    else:
                        # validate
                        valid_since = time.time()
                        visual_contexts = encoder.extract(visual_inputs)
                        # teacher forcing
                        outputs, _ = decoder(visual_contexts, caption_inputs, cap_lengths)
                        # # no teacher forcing
                        # outputs = decoder.sample(visual_contexts, cap_lengths)
                        loss = self.criterion(outputs, caption_targets)
                        log['valid_time'].append(time.time() - valid_since)
                        
                        # decode outputs
                        outputs = self._decode_outputs(outputs.max(1)[1], pack_info, cap_lengths, dictionary)
                        # save to candidates
                        for model_id, output in zip(model_ids, outputs):
                            if model_id not in candidates[phase].keys():
                                candidates[phase][model_id] = [output]
                            else:
                                candidates[phase][model_id].append(output)

                        # save log
                        log['valid_loss'].append(loss.data[0])
            
            # accumulate loss
            log['train_loss'] = np.mean(log['train_loss'])
            log['valid_loss'] = np.mean(log['valid_loss'])
            # evaluate bleu
            eval_since = time.time()
            train_blue, _ = capbleu.Bleu(4).compute_score(references["train"], candidates["train"])
            valid_blue, _ = capbleu.Bleu(4).compute_score(references["valid"], candidates["valid"])
            # evaluate cider
            train_cider, _ = capcider.Cider().compute_score(references["train"], candidates["train"])
            valid_cider, _ = capcider.Cider().compute_score(references["valid"], candidates["valid"])
            # # evaluate meteor
            # try:
            #     train_meteor, _ = capmeteor.Meteor().compute_score(references["train"], candidates["train"])
            #     valid_meteor, _ = capmeteor.Meteor().compute_score(references["valid"], candidates["valid"])
            # except Exception:
            #     train_meteor = 0
            #     valid_meteor = 0
            # evaluate rouge
            train_rouge, _ = caprouge.Rouge().compute_score(references["train"], candidates["train"])
            valid_rouge, _ = caprouge.Rouge().compute_score(references["valid"], candidates["valid"])
            log['eval_time'] = time.time() - eval_since
            
            # log
            log['train_blue_1'] = train_blue[0]
            log['train_blue_2'] = train_blue[1]
            log['train_blue_3'] = train_blue[2]
            log['train_blue_4'] = train_blue[3]
            log['valid_blue_1'] = valid_blue[0]
            log['valid_blue_2'] = valid_blue[1]
            log['valid_blue_3'] = valid_blue[2]
            log['valid_blue_4'] = valid_blue[3]
            log['train_cider'] = train_cider
            log['valid_cider'] = valid_cider
            # log['train_meteor'] = train_meteor
            # log['valid_meteor'] = valid_meteor
            log['train_rouge'] = train_rouge
            log['valid_rouge'] = valid_rouge

            # update report on tensorboard after every epoch
            writer.add_scalar("loss/train_loss", log['train_loss'], epoch_id)
            writer.add_scalar("loss/valid_loss", log['valid_loss'], epoch_id)
            writer.add_scalar("blue/train_blue_1", log['train_blue_1'], epoch_id)
            writer.add_scalar("blue/valid_blue_1", log['valid_blue_1'], epoch_id)
            writer.add_scalar("blue/train_blue_2", log['train_blue_2'], epoch_id)
            writer.add_scalar("blue/valid_blue_2", log['valid_blue_2'], epoch_id)
            writer.add_scalar("blue/train_blue_3", log['train_blue_3'], epoch_id)
            writer.add_scalar("blue/valid_blue_3", log['valid_blue_3'], epoch_id)
            writer.add_scalar("blue/train_blue_4", log['train_blue_4'], epoch_id)
            writer.add_scalar("blue/valid_blue_4", log['valid_blue_4'], epoch_id)
            writer.add_scalar("blue/train_cider", log['train_cider'], epoch_id)
            writer.add_scalar("blue/valid_cider", log['valid_cider'], epoch_id)
            # writer.add_scalar("blue/train_meteor", log['train_meteor'], epoch_id)
            # writer.add_scalar("blue/valid_meteor", log['valid_meteor'], epoch_id)
            writer.add_scalar("blue/train_rouge", log['train_rouge'], epoch_id)
            writer.add_scalar("blue/valid_rouge", log['valid_rouge'], epoch_id)

            log['epoch_time'].append(np.mean(time.time() - start))
            # show report
            if epoch_id % verbose == (verbose - 1):
                exetime_s = np.sum(log['epoch_time'])
                eta_s = exetime_s * (epoch - (epoch_id))
                eta_m = math.floor(eta_s / 60)
                print("---------------------epoch %d/%d----------------------" % (epoch_id, epoch))
                print("[Loss] train_loss: %f, valid_loss: %f" % (
                    log['train_loss'], 
                    log['valid_loss'])
                )
                print("[BLEU-1] train_blue: %f, valid_blue: %f" % (
                    log['train_blue_1'],
                    log['valid_blue_1'])
                )
                print("[BLEU-2] train_blue: %f, valid_blue: %f" % (
                    log['train_blue_2'],
                    log['valid_blue_2'])
                )
                print("[BLEU-3] train_blue: %f, valid_blue: %f" % (
                    log['train_blue_3'],
                    log['valid_blue_3'])
                )
                print("[BLEU-4] train_blue: %f, valid_blue: %f" % (
                    log['train_blue_4'],
                    log['valid_blue_4'])
                )
                print("[CIDEr] train_cider: %f, valid_cider: %f" % (
                    log['train_cider'],
                    log['valid_cider'])
                )
                # print("[METEOR] train_meteor: %f, valid_meteor: %f" % (
                #     log['train_meteor'],
                #     log['valid_meteor'])
                # )
                print("[ROUGE_L] train_rouge: %f, valid_rouge: %f" % (
                    log['train_rouge'],
                    log['valid_rouge'])
                )
                print("[Info]  forward_per_epoch: %fs\n[Info]  backward_per_epoch: %fs\n[Info]  valid_per_epoch: %fs" % (
                    np.sum(log['forward']), 
                    np.sum(log['backward']),
                    np.sum(log['valid_time']))
                )
                print("[Info]  eval_time: %fs" % ( 
                    np.mean(log['eval_time']))
                )
                print("[Info]  time_per_epoch: %fs\n[Info]  ETA: %dm %ds \n" % ( 
                    np.mean(log['epoch_time']),
                    eta_m,
                    eta_s - eta_m * 60)
                )
            
            # save log
            self.log[epoch_id] = log
            
            # save model
            torch.save(encoder, "models/encoder_checkpoint_%s.pth" % self.model_type)
            torch.save(decoder, "models/decoder_checkpoint_%s.pth" % self.model_type)

        # export scalar data to JSON for external processing
        writer.export_scalars_to_json("logs/all_scalars.json")
        writer.close()

                