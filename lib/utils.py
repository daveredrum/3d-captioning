import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def decode_outputs(sequence, cap_lengths, dictionary, phase):
    decoded = []
    if phase == "train":
        # get the indices for each predicted word
        _, indices = torch.max(sequence, 2)
        # chop the sequences according to their lengths
        unpadded_sequence = [indices[i][:int(cap_lengths.tolist()[i])].tolist() for i in range(cap_lengths.size(0))]
        # decode the indices
        for sequence in unpadded_sequence:
            temp = []
            for idx in sequence:
                try:
                    temp.append(dictionary[str(idx)])
                except Exception:
                    pass
            decoded.append(" ".join(temp))
    elif phase == "val":
        for i in range(len(sequence)):
            temp = []
            for j in range(len(sequence[i])):
                try:
                    temp.append(dictionary[str(sequence[i][j])])
                except Exception:
                    pass
            decoded.append(" ".join(temp))

    return decoded

# for model with attention
def decode_attention_outputs(sequence, cap_lengths, dictionary, phase):
    decoded = []
    if phase == "train":
        # get the indices for each predicted word
        _, indices = torch.max(sequence, 2)
        # chop the sequences according to their lengths
        unpadded_sequence = [indices[i][:int(cap_lengths.tolist()[i])-1].tolist() for i in range(cap_lengths.size(0))]
        # decode the indices
        for sequence in unpadded_sequence:
            temp = ['<START>']
            for idx in sequence:
                try:
                    temp.append(dictionary[str(idx)])
                except Exception:
                    pass
            decoded.append(" ".join(temp))
    elif phase == "val":
        for i in range(len(sequence)):
            temp = ['<START>']
            for j in range(len(sequence[i])):
                try:
                    temp.append(dictionary[str(sequence[i][j])])
                except Exception:
                    pass
            decoded.append(" ".join(temp))

    return decoded

def clip_grad_value_(optimizer, clip_value):
    '''
    in-place gradient clipping
    '''
    clip_value = float(clip_value)
    for group in optimizer.param_groups:
        for param in group['params']:
            try:
                param.grad.data.clamp_(-clip_value, clip_value)
            except AttributeError:
                pass

def decode_log_embedding(return_log):
    train_log = {
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
    val_log = {
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
    while not return_log.empty():
        data = return_log.get()
        train_log['total_loss'].append(data['train']['total_loss'])
        train_log['walker_loss_tst'].append(data['train']['walker_loss_tst'])
        train_log['walker_loss_sts'].append(data['train']['walker_loss_sts'])
        train_log['visit_loss_ts'].append(data['train']['visit_loss_ts'])
        train_log['visit_loss_st'].append(data['train']['visit_loss_st'])
        train_log['metric_loss_st'].append(data['train']['metric_loss_st'])
        train_log['metric_loss_tt'].append(data['train']['metric_loss_tt'])
        train_log['shape_norm_penalty'].append(data['train']['shape_norm_penalty'])
        train_log['text_norm_penalty'].append(data['train']['text_norm_penalty'])
        val_log['total_loss'].append(data['val']['total_loss'])
        val_log['walker_loss_tst'].append(data['val']['walker_loss_tst'])
        val_log['walker_loss_sts'].append(data['val']['walker_loss_sts'])
        val_log['visit_loss_ts'].append(data['val']['visit_loss_ts'])
        val_log['visit_loss_st'].append(data['val']['visit_loss_st'])
        val_log['metric_loss_st'].append(data['val']['metric_loss_st'])
        val_log['metric_loss_tt'].append(data['val']['metric_loss_tt'])
        val_log['shape_norm_penalty'].append(data['val']['shape_norm_penalty'])
        val_log['text_norm_penalty'].append(data['val']['text_norm_penalty'])
    
    # aggregate
    train_log['total_loss'] = np.mean(np.array(train_log['total_loss']), 0)
    train_log['walker_loss_tst'] = np.mean(np.array(train_log['walker_loss_tst']), 0)
    train_log['walker_loss_sts'] = np.mean(np.array(train_log['walker_loss_sts']), 0)
    train_log['visit_loss_ts'] = np.mean(np.array(train_log['visit_loss_ts']), 0)
    train_log['visit_loss_st'] = np.mean(np.array(train_log['visit_loss_st']), 0)
    train_log['metric_loss_st'] = np.mean(np.array(train_log['metric_loss_st']), 0)
    train_log['metric_loss_tt'] = np.mean(np.array(train_log['metric_loss_tt']), 0)
    train_log['shape_norm_penalty'] = np.mean(np.array(train_log['shape_norm_penalty']), 0)
    train_log['text_norm_penalty'] = np.mean(np.array(train_log['text_norm_penalty']), 0)
    val_log['total_loss'] = np.mean(np.array(val_log['total_loss']), 0)
    val_log['walker_loss_tst'] = np.mean(np.array(val_log['walker_loss_tst']), 0)
    val_log['walker_loss_sts'] = np.mean(np.array(val_log['walker_loss_sts']), 0)
    val_log['visit_loss_ts'] = np.mean(np.array(val_log['visit_loss_ts']), 0)
    val_log['visit_loss_st'] = np.mean(np.array(val_log['visit_loss_st']), 0)
    val_log['metric_loss_st'] = np.mean(np.array(val_log['metric_loss_st']), 0)
    val_log['metric_loss_tt'] = np.mean(np.array(val_log['metric_loss_tt']), 0)
    val_log['shape_norm_penalty'] = np.mean(np.array(val_log['shape_norm_penalty']), 0)
    val_log['text_norm_penalty'] = np.mean(np.array(val_log['text_norm_penalty']), 0)

    return train_log, val_log

def draw_curves_embedding(train_log, val_log, root):
    print("plotting training curves...")
    if not os.path.exists(os.path.join(root, 'curves')):
        os.mkdir(os.path.join(root, 'curves'))
    
    # settings
    plt.switch_backend("agg")
    fig = plt.gcf()
    fig.set_size_inches(16,8)
    epochs = train_log['total_loss'].shape[0]

    # plot total loss
    plt.plot(range(1, epochs + 1), train_log['total_loss'], label="train_loss")
    plt.plot(range(1, epochs + 1), val_log['total_loss'], label="val_loss")
    plt.xlabel('epoch')
    plt.ylabel('total_loss')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.savefig(os.path.join(root, 'curves', 'total_loss.png'), bbox_inches="tight")

    # plot walker loss
    fig.clf()
    fig.set_size_inches(16,16)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), train_log['walker_loss_tst'], label="train_walker_loss_tst")
    plt.plot(range(1, epochs + 1), val_log['walker_loss_tst'], label="val_walker_loss_tst")
    plt.xlabel('epoch')
    plt.ylabel('walker_loss_tst')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), train_log['walker_loss_sts'], label="train_walker_loss_sts")
    plt.plot(range(1, epochs + 1), val_log['walker_loss_sts'], label="val_walker_loss_sts")
    plt.xlabel('epoch')
    plt.ylabel('walker_loss_sts')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.savefig(os.path.join(root, 'curves', 'walker_loss.png'), bbox_inches="tight")

    # plot visit loss
    fig.clf()
    fig.set_size_inches(16,16)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), train_log['visit_loss_ts'], label="train_visit_loss_ts")
    plt.plot(range(1, epochs + 1), val_log['visit_loss_ts'], label="val_visit_loss_ts")
    plt.xlabel('epoch')
    plt.ylabel('visit_loss_ts')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), train_log['visit_loss_st'], label="train_visit_loss_st")
    plt.plot(range(1, epochs + 1), val_log['visit_loss_st'], label="val_visit_loss_st")
    plt.xlabel('epoch')
    plt.ylabel('visit_loss_st')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.savefig(os.path.join(root, 'curves', 'visit_loss.png'), bbox_inches="tight")

    # plot metric learning loss
    fig.clf()
    fig.set_size_inches(16,16)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), train_log['metric_loss_st'], label="train_metric_loss_st")
    plt.plot(range(1, epochs + 1), val_log['metric_loss_st'], label="val_metric_loss_st")
    plt.xlabel('epoch')
    plt.ylabel('metric_loss_st')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), train_log['metric_loss_tt'], label="train_metric_loss_tt")
    plt.plot(range(1, epochs + 1), val_log['metric_loss_tt'], label="val_metric_loss_tt")
    plt.xlabel('epoch')
    plt.ylabel('metric_loss_tt')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.savefig(os.path.join(root, 'curves', 'metric_loss.png'), bbox_inches="tight")

    # plot metric learning loss
    fig.clf()
    fig.set_size_inches(16,16)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), train_log['shape_norm_penalty'], label="train_shape_norm_penalty")
    plt.plot(range(1, epochs + 1), val_log['shape_norm_penalty'], label="val_shape_norm_penalty")
    plt.xlabel('epoch')
    plt.ylabel('shape_norm_penalty')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), train_log['text_norm_penalty'], label="train_text_norm_penalty")
    plt.plot(range(1, epochs + 1), val_log['text_norm_penalty'], label="val_text_norm_penalty")
    plt.xlabel('epoch')
    plt.ylabel('text_norm_penalty')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.savefig(os.path.join(root, 'curves', 'norm_penalty.png'), bbox_inches="tight")
    