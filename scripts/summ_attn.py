import argparse
import os
import math
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# HACK
import sys
sys.path.append(".")
from lib.configs import CONF

def summarize(model_id, attn_path_dict, output_path):
    plt.clf()
    fig = plt.gcf()
    num_img = len(attn_path_dict[model_id]) + 1
    num_col = 5
    num_row = math.ceil(num_img / num_col)
    img_width = 6
    fig.set_size_inches(img_width * num_col, img_width * num_row)
    for i in range(num_img):
        plt.subplot(num_row, num_col, i + 1)
        if i == 0:
            raw_model_img = Image.open(os.path.join(CONF.PATH.SHAPENET_ROOT.format(64), CONF.PATH.SHAPENET_IMG.format(model_id, model_id))).resize((224, 224))
            plt.imshow(raw_model_img)
            plt.xticks([])
            plt.yticks([])
            plt.text(0, 10, "<START>", fontsize=20, bbox={'facecolor': '#E0E0E0'})
        else:
            img = Image.open(attn_path_dict[model_id][i - 1]).resize((224, 224))
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            if i == num_img - 1:
                plt.text(0, 10, "<END>", fontsize=20, bbox={'facecolor': '#E0E0E0'})
            else:
                file_full_name = attn_path_dict[model_id][i - 1].split("/")[-1]
                file_name = os.path.splitext(file_full_name)[0]
                text = file_name.split("_")[1][1:-1]
                plt.text(0, 10, text, fontsize=20, bbox={'facecolor': '#E0E0E0'})

    plt.savefig(os.path.join(output_path, "{}.png".format(model_id)), bbox_inches="tight")

def filter_png(attn_root):
    model_list = os.listdir(attn_root)
    filtered = {model_id: None for model_id in model_list}
    for model_id in model_list:
        temp = []
        for file_full_name in os.listdir(os.path.join(attn_root, model_id)):
            _, file_ext = os.path.splitext(file_full_name)
            if file_ext == ".png":
                temp.append(os.path.join(attn_root, model_id, file_full_name))
       
        # print(temp.split("/")[-1].split(".")[0].split("_")[0])
        temp.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[0]))
        filtered[model_id] = temp

    return filtered

def main(args):
    attn_root = os.path.join(CONF.PATH.OUTPUT_CAPTION, args.path, "vis")
    attn_path_dict = filter_png(attn_root)
    output_path = os.path.join(CONF.PATH.ROOT, "outputs/summaries/attn")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        for file in os.listdir(output_path):
            os.remove(os.path.join(output_path, file))

    plt.switch_backend("agg")
    model_ids = list(attn_path_dict.keys())
    for i, model_id in enumerate(model_ids):
        summarize(model_id, attn_path_dict, output_path)
        print("saved summary for {}, {} left".format(model_id, len(model_ids) - i - 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the caption root folder")
    args = parser.parse_args()
    main(args)