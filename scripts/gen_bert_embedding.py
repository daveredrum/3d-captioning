import os
import h5py
import argparse
import pickle
import time
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# HACK
import sys
sys.path.append(".")
from lib.configs import CONF

def get_eta(start, end, num_left):
    exe_s = end - start
    eta_s = exe_s * num_left
    eta = {'h': 0, 'm': 0, 's': 0}
    if eta_s < 60:
        eta['s'] = int(eta_s)
    elif eta_s >= 60 and eta_s < 3600:
        eta['m'] = int(eta_s / 60)
        eta['s'] = int(eta_s % 60)
    else:
        eta['h'] = int(eta_s / (60 * 60))
        eta['m'] = int(eta_s % (60 * 60) / 60)
        eta['s'] = int(eta_s % (60 * 60) % 60)

    return eta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="0/1/2/3")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # load data
    print("loading data...")
    raw_data = {
        "train": pickle.load(open(os.path.join(CONF.PATH.PROC_DATA_ROOT, CONF.PATH.SPLIT_NAME.format("train")), 'rb')),
        "val": pickle.load(open(os.path.join(CONF.PATH.PROC_DATA_ROOT, CONF.PATH.SPLIT_NAME.format("val")), 'rb')),
        "test": pickle.load(open(os.path.join(CONF.PATH.PROC_DATA_ROOT, CONF.PATH.SPLIT_NAME.format("test")), 'rb'))
    }

    # set bert
    print("loading BERT...")
    tokenizer = BertTokenizer.from_pretrained(CONF.BERT.MODEL)
    bert = BertModel.from_pretrained(CONF.BERT.MODEL).cuda()
    bert.eval()

    # set database
    database = h5py.File(CONF.PATH.BERT_EMBEDDING, "w", libver='latest')
    for phase in ["train", "val", "test"]:
        print("processing {}...".format(phase))
        dataset = database.create_dataset(phase, (len(raw_data[phase]["data"]), CONF.BERT.DIM), dtype="float")
        for i, item in enumerate(raw_data[phase]["data"]):
            start = time.time()
            tokenized_text = tokenizer.tokenize(" ".join(item[2]))
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens]).cuda()
            _, outputs = bert(tokens_tensor)
            dataset[i] = outputs.squeeze().data.cpu().numpy()
            eta = get_eta(start, time.time(), len(raw_data[phase]["data"]) - i - 1)
            if (i + 1) % 100 == 0:
                print("processed {}, {} left, ETA: {}h {}m {}s".format(i + 1, len(raw_data[phase]["data"]) - i - 1, eta['h'], eta['m'], eta['s']))
