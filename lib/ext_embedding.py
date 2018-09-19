import argparse
import pickle
import time
import math
import torch
import os
from torch.utils.data import Dataset, DataLoader

# HACK
import sys
sys.path.append(".")
from lib.data_embedding import Embedding, EmbeddingDataset, collate_embedding
from lib.configs import CONF
from model.encoder_shape import ShapeEncoder
from model.encoder_text import TextEncoder
from model.encoder_attn import SelfAttnShapeEncoder, SelfAttnTextEncoder

def parse_path(path):
    setting_list = path.split("_")
    print(setting_list)
    settings = {
        'dataset': setting_list[0].split("]")[1],
        'resolution': setting_list[1][1:],
        'train_size': setting_list[2][3:],
        'learning_rate': setting_list[3][2:],
        'weight_decay': setting_list[4][2:],
        'epoch': setting_list[5][1:],
        'batch_size': setting_list[6][2:],
        'attention_type': setting_list[7]
    }

    return settings

def get_dataset(settings):
    data = Embedding(
        [
            pickle.load(open("data/{}_split_train.p".format(settings['dataset']), 'rb')),
            pickle.load(open("data/{}_split_val.p".format(settings['dataset']), 'rb')),
            pickle.load(open("data/{}_split_test.p".format(settings['dataset']), 'rb'))
        ],
        [
            -1,
            -1,
            -1
        ],
        None,
        False
    )
    dataset = {
        'train': EmbeddingDataset(
            data.train_data, 
            data.train_idx2label, 
            data.train_label2idx, 
            settings['resolution']
        ),
        'val': EmbeddingDataset(
            data.val_data, 
            data.val_idx2label, 
            data.val_label2idx, 
            settings['resolution']
        ),
        'test': EmbeddingDataset(
            data.test_data, 
            data.test_idx2label, 
            data.test_label2idx, 
            settings['resolution']
        )
    }

    return data, dataset

def get_dataloader(dataset, settings):
    dataloader = {
        'train': DataLoader(
            dataset['train'], 
            batch_size=int(settings['batch_size']),  
            collate_fn=collate_embedding
        ),
        'val': DataLoader(
            dataset['val'], 
            batch_size=int(settings['batch_size']),  
            collate_fn=collate_embedding
        ),
        'test': DataLoader(
            dataset['test'], 
            batch_size=int(settings['batch_size']),  
            collate_fn=collate_embedding
        )
    }

    return dataloader

def get_model(settings, data, root):
    if settings['attention_type'] == 'noattention' or settings['attention_type'] == 'text2shape':
        print("initializing naive models...")
        shape_encoder = ShapeEncoder(is_final=True).cuda()
        text_encoder = TextEncoder(data.dict_idx2word.__len__()).cuda()
    else:
        print("initializing {} models...".format(settings['attention_type']))
        shape_encoder = SelfAttnShapeEncoder(settings['attention_type'], is_final=True).cuda()
        text_encoder = SelfAttnTextEncoder(data.dict_idx2word.__len__()).cuda()

    # load
    shape_encoder.load_state_dict(torch.load(os.path.join(root, "models/shape_encoder.pth")))
    shape_encoder.eval()
    text_encoder.load_state_dict(torch.load(os.path.join(root, "models/text_encoder.pth")))
    text_encoder.eval()

    return shape_encoder, text_encoder

def get_embedding(shape_encoder, text_encoder, dataloader, data):
    '''
        embedding = {
            'train': {
                <model_id>: {
                    'shape_embedding': (
                        <shape_embedding>,
                        <shape_interm_embedding>
                    ),
                    'text_embedding': (
                        <caption>,
                        <text_embedding>
                    )
                }
                ...
            }
            'val': {...}
            'test': {...}
        }
    '''
    # extract
    embedding = {
        'train': {},
        'val': {},
        'test': {}
    }
    for phase in ['train', 'val', 'test']:
        print("\nstart extracting {}...\n".format(phase))
        offset = 0
        total_iter = len(dataloader[phase])
        for iter_id, (model_id, shape, text, _, _, _) in enumerate(dataloader[phase]):
            start = time.time()
            shape = shape.cuda()
            text = text.cuda()
            if isinstance(shape_encoder, SelfAttnShapeEncoder) and isinstance(text_encoder, SelfAttnTextEncoder):
                (shape_embedding, shape_interm_embedding), _ = shape_encoder(shape)
                text_embedding, _ = text_encoder(text)
            else:
                shape_embedding, shape_interm_embedding = shape_encoder(shape)
                text_embedding = text_encoder(text)
            
            # dump
            for i in range(len(model_id)):
                cap = " ".join([data.dict_idx2word[str(idx.item())] for idx in text[i] if idx.item() != 0])
                if model_id[i] in embedding[phase].keys():
                    embedding[phase][model_id[i]]['text_embedding'].append(
                        (
                            cap,
                            text_embedding[i].data.cpu().numpy()
                        )
                    ) 
                else:
                    embedding[phase][model_id[i]] = {
                        'shape_embedding': (
                            shape_embedding[i].data.cpu().numpy(),
                            shape_interm_embedding[i].data.cpu().numpy()
                        ),
                        'text_embedding': [
                            (
                                cap,
                                text_embedding[i].data.cpu().numpy()
                            )
                        ]
                    }

            # report
            offset += len(model_id)
            exe_s = time.time() - start
            eta_s = exe_s * (total_iter - (iter_id + 1))
            eta_m = math.floor(eta_s / 60)
            eta_s = math.floor(eta_s % 60)
            print("extracted: {}/{}, ETA: {}m {}s".format(offset, len(getattr(data, "{}_data".format(phase))), eta_m, int(eta_s)))

    return embedding

def save_embedding(embedding, root):
    print("\nsaving embedding...")
    embedding_root = os.path.join(root, "embedding")
    embedding_name = os.path.join(root, "embedding", "embedding.p")
    if not os.path.exists(embedding_root):
        os.mkdir(embedding_root)
    pickle.dump(embedding, open(embedding_name, 'wb'))

def main(args):
    # setting
    print("\npreparing...\n")
    root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    settings = parse_path(root)
    data, dataset = get_dataset(settings)
    dataloader = get_dataloader(dataset, settings)
    shape_encoder, text_encoder = get_model(settings, data, root)
    embedding = get_embedding(shape_encoder, text_encoder, dataloader, data)
    save_embedding(embedding, root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the folder")
    parser.add_argument("--gpu", type=str, default="2", help="choose gpu")
    args = parser.parse_args()
    main(args)