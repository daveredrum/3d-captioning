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
    return return_log['train'], return_log['val'], return_log['eval']

def draw_curves_embedding(train_log, val_log, eval_log, root):
    print("plotting training curves...")
    if not os.path.exists(os.path.join(root, 'curves')):
        os.mkdir(os.path.join(root, 'curves'))
    
    # settings
    plt.switch_backend("agg")
    fig = plt.gcf()
    fig.set_size_inches(16,8)
    epochs = len(train_log['total_loss'])

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

    # plot eval scores
    fig.clf()
    fig.set_size_inches(16,32)
    plt.subplot(4, 1, 1)
    plt.plot(range(1, epochs + 1), eval_log['total_score_t2s'], label="total_score_t2s")
    plt.plot(range(1, epochs + 1), eval_log['total_score_s2t'], label="total_score_s2t")
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(range(1, epochs + 1), eval_log['recall_1_t2s'], label="RR@1_t2s")
    plt.plot(range(1, epochs + 1), eval_log['recall_1_s2t'], label="RR@1_s2t")
    plt.xlabel('epoch')
    plt.ylabel('RR@1')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(range(1, epochs + 1), eval_log['recall_5_t2s'], label="RR@5_t2s")
    plt.plot(range(1, epochs + 1), eval_log['recall_5_s2t'], label="RR@5_s2t")
    plt.xlabel('epoch')
    plt.ylabel('RR@5')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(range(1, epochs + 1), eval_log['ndcg_5_t2s'], label="NDCG@5_t2s")
    plt.plot(range(1, epochs + 1), eval_log['ndcg_5_s2t'], label="NDCG@5_s2t")
    plt.xlabel('epoch')
    plt.ylabel('NDCG@5')
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.savefig(os.path.join(root, 'curves', 'scores.png'), bbox_inches="tight")
    
    
def report_best(best):
    print("------------------------best------------------------")
    print("[Score] epoch: %d" % (
        best['epoch']
    ))
    print("[Score] total_scores: %f" % (
        best['total_score']
    ))
    print("[Score] recall_1_t2s: %f, recall_1_s2t: %f" % (
        best['recall_1_t2s'],
        best['recall_1_s2t']
    ))
    print("[Score] recall_5_t2s: %f, recall_5_s2t: %f" % (
        best['recall_5_t2s'],
        best['recall_5_s2t']
    ))
    print("[Score] ndcg_5_t2s: %f, ndcg_5_s2t: %f" % (
        best['ndcg_5_t2s'],
        best['ndcg_5_s2t']
    ))
    print("[Loss]  total_loss: %f" % (
        best['total_loss']
    ))
    print("[Loss]  walker_loss_tst: %f, walker_loss_sts: %f" % (
        best['walker_loss_tst'],
        best['walker_loss_sts']
    ))
    print("[Loss]  visit_loss_ts: %f, visit_loss_st: %f" % (
        best['visit_loss_ts'],
        best['visit_loss_st']
    ))
    print("[Loss]  metric_loss_st: %f, metric_loss_tt: %f" % (
        best['metric_loss_st'],
        best['metric_loss_tt']
    ))
    print("[Loss]  shape_norm_penalty: %f, text_norm_penalty: %f\n" % (
        best['shape_norm_penalty'],
        best['text_norm_penalty']
    ))