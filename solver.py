import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

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
            self.log[epoch] = log

class DecoderSolver():
    def __init__(self, optimizer, criterion, cuda_flag=True):
        self.optimizer = optimizer
        self.criterion = criterion
        self.cuda_flag = cuda_flag
        self.log = {}
    
    def _create_chunks(self, caption_lists, size):
        for i in range(0, len(caption_lists), size):
            yield caption_lists[i: i+size]

    def train(self, model, training_pairs, epoch):
        # training_pairs is a list of (visual_context, caption)
        # visual_context and caption are both numpy arrays
        for epoch_id in range(epoch):
            log = []
            for visual_context, caption in training_pairs:
                # prepare the visual context vector before feeding into the model
                caption_size = len(caption)
                loss = 0
                if self.cuda_flag:
                    visual_inputs = Variable(torch.from_numpy(visual_context)).view(1, 1, visual_context.shape[0]).cuda()
                    # initialize the hidden states h and c
                    hiddens = (visual_inputs, Variable(torch.zeros(*(list(visual_inputs.size())))).cuda())
                else:
                    visual_inputs = Variable(torch.from_numpy(visual_context)).view(1, 1, visual_context.shape[0])
                    # initialize the hidden states h and c
                    hiddens = (visual_inputs, Variable(torch.zeros(*(list(visual_inputs.size())))))
                for text_id in range(caption_size - 2):
                    if self.cuda_flag:
                        caption_inputs = Variable(torch.tensor(caption[text_id])).view(1, 1).cuda()
                        caption_targets = Variable(torch.tensor(caption[text_id + 1])).cuda()
                    else:
                        caption_inputs = Variable(torch.tensor(caption[text_id])).view(1, 1)
                        caption_targets = Variable(torch.tensor(caption[text_id + 1]))
                    # feed the model and compute the loss
                    outputs, hiddens = model(caption_inputs, hiddens)
                    loss += self.criterion(outputs.view(1, -1), caption_targets)
                loss.backward()
                self.optimizer.step()
                log.append(loss.data[0])
            print("[epoch %d] training loss: %f" % (epoch_id + 1, np.mean(log)))

                