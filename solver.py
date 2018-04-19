import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence

# only for encoder offline training 
class EncoderSolver():
    def __init__(self, optimizer, criterion, cuda_flag=True):
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda_flag = cuda_flag
        self.log = {}

    def train(self, model, dataloader, epoch):
        for epoch_id in range(epoch):
            log = {
                'train_loss': [],
                'train_acc': [],
                'valid_loss': [],
                'valid_acc': []    
            }
            for phase in ['train', 'valid']:
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
            # show status at the end of every epoch 
            print("[epoch %d]: train_loss: %f, train_acc: %f, valid_loss: %f, valid_acc: %f" % (
                epoch_id + 1, 
                np.mean(log['train_loss']), 
                np.mean(log['train_acc']),
                np.mean(log['valid_loss']), 
                np.mean(log['valid_acc'])
            ))
            # save log
            self.log[epoch_id] = log

# only for decoder offline training
# extracted visual contexts required
class DecoderSolver():
    def __init__(self, optimizer, criterion, cuda_flag=True):
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda_flag = cuda_flag
        self.log = {}
    
    def _create_chunks(self, caption_lists, size):
        for i in range(0, len(caption_lists), size):
            yield caption_lists[i: i+size]

    def train(self, model, dataloader, epoch, verbose):
        # training_pairs is a list of (visual_context, caption)
        # visual_context and caption are both numpy arrays
        for epoch_id in range(epoch):
            log = {
                'train_loss': [],
                'valid_loss': []   
            }
            for phase in ["train", "valid"]:
                for visuals, captions, cap_lengths in dataloader[phase]:
                    self.optimizer.zero_grad()
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
                    outputs = model(visual_inputs, caption_inputs, cap_lengths)
                    loss = self.criterion(outputs, caption_targets)
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()
                        log['train_loss'].append(loss.data[0])
                    else:
                        log['valid_loss'].append(loss.data[0])
            # show report
            if epoch_id % verbose == (verbose - 1):
                print("[epoch %d/%d] train_loss: %f, valid_loss: %f" % (
                    epoch_id + 1,
                    epoch, 
                    np.mean(log['train_loss']), 
                    np.mean(log['valid_loss'])
                    ))
            # save log
            self.log[epoch_id] = log
                