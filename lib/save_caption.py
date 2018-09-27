import os
import argparse
import pickle
import torch
import time

# HACK
import sys
sys.path.append(".")
import lib.capeval.bleu.bleu as capbleu
import lib.capeval.cider.cider as capcider
from model.encoder_shape import *
from model.encoder_attn import *
from model.decoders_caption import *
from lib.configs import CONF
from lib.data_caption import *
from lib.utils import decode_outputs, decode_attention_outputs

def parse_path(path):
    print("parsing settings...")
    setting_list = path.split("_")
    settings = {
        'dataset': setting_list[0].split("]")[1],
        'encoder_type': setting_list[1],
        'attention_type': setting_list[2]
    }

    return settings

def get_dataset(embedding_root):
    print("getting dataset...")
    embedding_path = os.path.join(embedding_root, "embedding", "embedding.p")
    data = PretrainedEmbeddings(pickle.load(open(embedding_path, 'rb')))
    dataset = {
        'train': CaptionDataset(data.train_text, data.train_shape),
        'val': CaptionDataset(data.val_text, data.val_shape),
        'test': CaptionDataset(data.test_text, data.test_shape)
    }

    return data, dataset

def get_dataloader(dataset, settings, batch_size):
    print("building dataloader...")
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=batch_size, collate_fn=collate_ec),
        'val': DataLoader(dataset['val'], batch_size=batch_size, collate_fn=collate_ec),
        'test': DataLoader(dataset['test'], batch_size=batch_size, collate_fn=collate_ec)
    }

    return dataloader

def get_model(settings, caption_root):
    if settings['attention_type'] == 'fc':
        print("initializing FC decoder...")
    else:
        print("initializing {} decoder...".format(settings['attention_type']))

    # load
    encoder = torch.load(os.path.join(caption_root, "models/encoder.pth")).cuda()
    decoder = torch.load(os.path.join(caption_root, "models/decoder.pth")).cuda()
    encoder.eval()
    decoder.eval()

    return encoder, decoder

def get_reference(data):
    print("building caption references...")
    references = {
        'train': data.train_ref,
        'val': data.val_ref,
        'test': data.test_ref
    }
    # load vocabulary
    dict_idx2word = data.dict_idx2word

    return references, dict_idx2word

def generate(encoder, decoder, dataloader, dict_idx2word, references, caption_root):
    print("generating captions...")
    outputs = {
        'train': {},
        'val': {},
        'test': {}
    }
    for phase in ["train", "val", "test"]:
        print("\ngenerating {}...".format(phase))
        for batch_id, (model_ids, captions, embeddings, embeddings_interm, lengths) in enumerate(dataloader[phase]):
            start = time.time()
            if isinstance(decoder, Decoder):
                visual_inputs = embeddings.cuda()
            else:
                visual_inputs = embeddings_interm.cuda()
            caption_inputs = captions[:, :-1].cuda()
            cap_lengths = lengths.cuda()
            visual_contexts = encoder(visual_inputs)
            max_length = int(cap_lengths[0].item()) + 10
            if isinstance(decoder, Decoder):
                candidates = decoder.beam_search(visual_contexts, 1, max_length)
                candidates = decode_outputs(candidates, None, dict_idx2word, "val")
            else:
                candidates = decoder.beam_search(visual_contexts, caption_inputs, 1, max_length)
                candidates = decode_attention_outputs(candidates, None, dict_idx2word, "val")
            for model_id, candidate in zip(model_ids, candidates):
                ref = {model_id: references[phase][model_id]}
                cand = {model_id: [candidate]}
                bleu = capbleu.Bleu(4).compute_score(ref, cand)[0]
                cider = capcider.Cider().compute_score(ref, cand)[0]
                output = (
                    candidate,
                    bleu,
                    cider
                )
                if model_id not in outputs.keys():
                    outputs[phase][model_id] = [output]
                else:
                    outputs[phase][model_id].append(output)
            
            exe_s = time.time() - start
            eta_s = exe_s * (len(dataloader[phase]) - (batch_id + 1))
            eta_m = math.floor(eta_s / 60)
            eta_s = math.floor(eta_s % 60)
            print("generated: {}/{}, ETA: {}m {}s".format(batch_id + 1, len(dataloader[phase]), eta_m, int(eta_s)))

    # save results
    print("\nsaving captions...")
    output_path = os.path.join(caption_root,  "caption.p")
    pickle.dump(outputs, open(output_path, 'wb'))

def main(args):
    # setting
    print("\npreparing data...")
    caption_root = os.path.join(CONF.PATH.OUTPUT_CAPTION, args.caption)
    embedding_root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.embedding)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    settings = parse_path(caption_root)
    data, dataset = get_dataset(embedding_root)
    dataloader = get_dataloader(dataset, settings, args.batch_size)
    references, dict_idx2word = get_reference(data)
    encoder, decoder = get_model(settings, caption_root)
    generate(encoder, decoder, dataloader, dict_idx2word, references, caption_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption", type=str, default=None, help="path to the captioning")
    parser.add_argument("--embedding", type=str, default=None, help="path to the embedding")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--gpu", type=str, default='2', help="specify the graphic card")
    args = parser.parse_args()
    main(args)