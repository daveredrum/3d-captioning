import os
import pickle
import argparse
from lib.configs import CONF
import subprocess
import matplotlib.pyplot as plt
from lib.data_embedding import Shapenet
from PIL import Image

def _check_rendered(path):
    flag = False
    if os.path.exists(path) and len(os.listdir(path)) == CONF.TRAIN.MAX_LENGTH:
        flag = True
    
    return flag

def summarize(root, tempo_attn, dict):
    plt.switch_backend("agg")
    for attn_idx, (key, item) in enumerate(tempo_attn.items()):
        mask, text = item
        fig = plt.figure(dpi=100)
        fig.set_size_inches(16, 12)
        fig.set_facecolor('white')

        # check if rendered
        if _check_rendered(os.path.join(root, "attention", key)):
            print("not rendered and skipped: {}, {} left".format(key, len(tempo_attn) - attn_idx - 1))
            continue

        # model image
        plt.subplot2grid((4, 6), (0, 0))
        model_id = key.split('-')[0]
        img_path = os.path.join(CONF.PATH.SHAPENET_ROOT.format(64), CONF.PATH.SHAPENET_IMG.format(model_id, model_id))
        img = Image.open(img_path)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

        # attention distribution
        plt.subplot2grid((4, 6), (0, 1), colspan=5)
        cap = [dict[str(idx)] for idx in text[0]]
        plt.bar(range(CONF.TRAIN.MAX_LENGTH), mask, tick_label=cap, edgecolor='none')
        for i, spine in enumerate(plt.gca().spines.values()):
            if i == 2:
                continue
            spine.set_visible(False)
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)

        # attended parts
        for i in range(CONF.TRAIN.MAX_LENGTH):
            plt.subplot2grid((4, 6), (i // 6 + 1, i % 6))
            img = Image.open(os.path.join(root, "attention", key, "{}.png".format(i)))
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.text(35, 95, cap[i], fontsize=12, bbox={'facecolor': '#e1e1e1', 'edgecolor': 'k'})
            
        plt.savefig(os.path.join(root, "vis", "{}.png".format(key)), bbox_inches="tight")
        plt.close()

        print("generated summary for model: {}, {} left".format(key, len(tempo_attn) - attn_idx - 1))

def main(args):
    root = os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path)
    tempo_attn = pickle.load(open(os.path.join(CONF.PATH.OUTPUT_EMBEDDING, args.path, "attention", "attn_mask.p"), 'rb'))
    shapenet = Shapenet(
        [
            pickle.load(open("data/shapenet_split_train.p", 'rb')),
            pickle.load(open("data/shapenet_split_val.p", 'rb')),
            pickle.load(open("data/shapenet_split_test.p", 'rb'))
        ],
        [-1] * 3,
        1,
        False
    )
    # render nrrd
    print("start generating summaries...\n")
    if not os.path.exists(os.path.join(root, "vis")):
        os.mkdir(os.path.join(root, "vis"))
    summarize(root, tempo_attn, shapenet.dict_idx2word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    args = parser.parse_args()
    main(args)