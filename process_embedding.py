import pickle
import os
import argparse
import configs

def main(args):
    if args.phases:
        phases = [args.phases]
    else:
        phases = ["train", "val", "test"]
    for phase in phases:
        print("phase:", phase)
        print()
        embeddings = pickle.load(open(os.path.join(configs.EMBEDDING_ROOT, configs.EMBEDDING_PRETRAINED.format(phase)), 'rb'))
        embeddings = {item[2]: item[3] for item in embeddings['caption_embedding_tuples']}
        with open(configs.EMBEDDING_PROCESSED.format(phase), 'wb') as f:
            pickle.dump(f, embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=str, default=None, help="train/val/test/None")
    args = parser.parse_args()
    main(args)