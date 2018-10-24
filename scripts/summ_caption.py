import argparse
import pickle
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# HACK
import sys
sys.path.append(".")
from lib.configs import CONF

def get_data():
    assert CONF.EVAL.COMP_CAP_A.keys() == CONF.EVAL.COMP_CAP_B.keys()
    method_list = list(CONF.EVAL.COMP_CAP_A.keys())
    data_method_a, data_method_b = {}, {}
    for method in method_list:
        data_method_a[method] = pickle.load(open(os.path.join(CONF.PATH.OUTPUT_CAPTION, CONF.EVAL.COMP_CAP_A[method], "caption.p"), 'rb'))
        data_method_b[method] = pickle.load(open(os.path.join(CONF.PATH.OUTPUT_CAPTION, CONF.EVAL.COMP_CAP_B[method], "caption.p"), 'rb'))

    return data_method_a, data_method_b

def summarize(root, data_method_a, data_method_b, method_list, model_id):
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(18, 5)
    widths = [0.2, 0.09, 0.07, 0.5, 0.06, 0.06]


    gs_img = gridspec.GridSpec(12, 1)
    gs_img.update(left=0, right=widths[0])
    ax = plt.subplot(gs_img[:, :])
    img = Image.open(os.path.join(CONF.PATH.SHAPENET_ROOT.format(CONF.EVAL.RESOLUTION), CONF.PATH.SHAPENET_IMG.format(model_id, model_id))).resize((200, 200))
    plt.imshow(img)
    plt.axis("off")

    
    gs_emb = gridspec.GridSpec(12, 1)
    gs_emb.update(left=widths[0], right=sum(widths[:2]))
    gs_method = gridspec.GridSpec(12, 1)
    gs_method.update(left=sum(widths[:2]), right=sum(widths[:3]))
    gs_cap = gridspec.GridSpec(12, 1)
    gs_cap.update(left=sum(widths[:3]), right=sum(widths[:4]))
    gs_bleu = gridspec.GridSpec(12, 1)
    gs_bleu.update(left=sum(widths[:4]), right=sum(widths[:5]))
    gs_cider = gridspec.GridSpec(12, 1)
    gs_cider.update(left=sum(widths[:5]), right=sum(widths[:6]))
    for i, method in enumerate(method_list):
        # first row
        ax = plt.subplot(gs_emb[3 * i, 0])
        plt.text(0, 0, "[noattn]", fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_method[3 * i, 0])
        plt.text(0, 0, "[{}]".format(method), fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_cap[3 * i, 0])
        cap = data_method_a[method][CONF.EVAL.EVAL_DATASET][model_id][0].split(" ")
        cap = " ".join(cap[1:-1])
        plt.text(0, 0, cap, fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_bleu[3 * i, 0])
        plt.text(0, 0, str(format(data_method_a[method][CONF.EVAL.EVAL_DATASET][model_id][1][3] * 100, '.3f')), fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_cider[3 * i, 0])
        plt.text(0, 0, str(format(data_method_a[method][CONF.EVAL.EVAL_DATASET][model_id][2] * 100, '.3f')), fontsize=15)
        plt.axis("off")

        # second row
        ax = plt.subplot(gs_emb[3 * i + 1, 0])
        plt.text(0, 0, "[SSAM]", fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_method[3 * i + 1, 0])
        plt.text(0, 0, "[{}]".format(method), fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_cap[3 * i + 1, 0])
        cap = data_method_b[method][CONF.EVAL.EVAL_DATASET][model_id][0].split(" ")
        cap = " ".join(cap[1:-1])
        plt.text(0, 0, cap, fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_bleu[3 * i + 1, 0])
        plt.text(0, 0, str(format(data_method_b[method][CONF.EVAL.EVAL_DATASET][model_id][1][3] * 100, '.3f')), fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_cider[3 * i + 1, 0])
        plt.text(0, 0, str(format(data_method_b[method][CONF.EVAL.EVAL_DATASET][model_id][2] * 100, '.3f')), fontsize=15)
        plt.axis("off")

        # blank row
        ax = plt.subplot(gs_emb[3 * i + 2, 0])
        plt.text(0, 0, " ", fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_method[3 * i + 2, 0])
        plt.text(0, 0, " ", fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_cap[3 * i + 2, 0])
        plt.text(0, 0, " ", fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_bleu[3 * i + 2, 0])
        plt.text(0, 0, " ", fontsize=15)
        plt.axis("off")

        ax = plt.subplot(gs_cider[3 * i + 2, 0])
        plt.text(0, 0, " ", fontsize=15)
        plt.axis("off")

    plt.savefig(os.path.join(root, "{}.png".format(model_id)), bbox_inches="tight")
    

def generate_vis(data_method_a, data_method_b):
    method_list = list(data_method_a.keys())
    all_model_ids = list(data_method_a[method_list[0]][CONF.EVAL.EVAL_DATASET].keys())
    chosen_model_ids = random.choices(all_model_ids, k=CONF.EVAL.NUM_CHOSEN)
    root = os.path.join(CONF.PATH.ROOT, "outputs/summaries/caption")
    if not os.path.exists(root):
        os.mkdir(root)
    else:
        for file in os.listdir(root):
            os.remove(os.path.join(root, file))
    
    plt.switch_backend("agg")
    for i, model_id in enumerate(chosen_model_ids):
        summarize(root, data_method_a, data_method_b, method_list, model_id)
        print("saved summary for {}, {} left".format(model_id, len(chosen_model_ids) - i - 1))

def main():
    data_method_a, data_method_b = get_data()
    generate_vis(data_method_a, data_method_b)

if __name__ == '__main__':
    main()