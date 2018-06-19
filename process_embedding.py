import pickle
import os
import argparse
import configs

def main(args):
    '''
    generate processed captions with embeddings in 'caption_embedding_tuples'
    '''
    if args.phases:
        phases = [args.phases]
    else:
        phases = ["train", "val", "test"]
    for phase in phases:
        print("phase:", phase)
        print()
        split = pickle.load(open(os.path.join(configs.SPLIT_ROOT, configs.SPLIT_NAME.format(phase)), 'rb'))['caption_tuples']
        pretrained = pickle.load(open(os.path.join(configs.PRETRAINED_ROOT, configs.PRETRAINED_SHAPE_EMBEDDING.format(phase)), 'rb'))
        pretrained = {item[2]: item[3] for item in pretrained['caption_embedding_tuples']}
        embeddings = {
            'caption_embedding_tuples': [(item[2], item[0], pretrained[item[2]]) for item in split]
        }
        
        with open(configs.PROCESSED_SHAPE_EMBEDDING.format(phase), 'wb') as f:
            pickle.dump(embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=str, default=None, help="train/val/test/None")
    args = parser.parse_args()
    main(args)