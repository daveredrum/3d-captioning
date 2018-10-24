import pickle
import os
import argparse
import torch
import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image

# HACK
import sys
sys.path.append(".")
from lib.configs import CONF

def get_path(args):
    print("\nparsing path...\n")
    path = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path, "embedding", "embedding.p")

    return path

def parse_emb(path, args):
    print("parsing embedding...\n")
    raw_embedding = pickle.load(open(path, "rb"))[args.phase]
    model_ids = list(raw_embedding.keys())
    shape_ids = model_ids
    shape_embeddings = np.concatenate([raw_embedding[model_id]["shape_embedding"][0][np.newaxis,: ] for model_id in model_ids], axis=0)
    text_ids = [model_id for model_id in model_ids for _ in raw_embedding[model_id]["text_embedding"]]
    text_embeddings = np.concatenate([embedding[1][np.newaxis,: ] for model_id in model_ids for embedding in raw_embedding[model_id]["text_embedding"]], axis=0)

    return shape_ids, torch.FloatTensor(shape_embeddings), text_ids, torch.FloatTensor(text_embeddings)

def get_image(ids):
    print("accessing images...\n")
    images = []
    for id in ids:
        raw_img = Image.open(os.path.join(CONF.PATH.SHAPENET_ROOT.format(64), CONF.PATH.SHAPENET_IMG.format(id, id))).resize((128, 128))
        img = Image.new("RGB", raw_img.size, (255, 255, 255))
        img.paste(raw_img, mask=raw_img.split()[3])
        img = np.transpose(np.array(img).astype(float), (2, 0, 1))
        img = img[np.newaxis, :]
        img /= 255.
        images.append(img)
    
    images = np.concatenate(images, axis=0)

    return torch.FloatTensor(images)

def vis(ids, embeddings, images, args):
    print("visualizing embeddings...\n")
    writer = SummaryWriter(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, "{}/tensorboard".format(args.path)))
    writer.add_embedding(embeddings, metadata=ids, label_img=images)
    writer.close()

def main(args):
    embedding_path = get_path(args)
    shape_ids, shape_embeddings, text_ids, text_embeddings = parse_emb(embedding_path, args)
    if args.mode == 's':
        images = get_image(shape_ids)
        vis(shape_ids, shape_embeddings, images, args)
    elif args.model == 't':
        images = get_image(text_ids)
        vis(text_ids, text_embeddings, images, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--phase", type=str, default="val", help="train/val/test")
    parser.add_argument("--mode", type=str, help="s/t")
    args = parser.parse_args()
    main(args)
