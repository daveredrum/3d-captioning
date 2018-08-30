import os
import argparse
from lib.configs import CONF
import subprocess


def render(root):
    assert os.path.exists(os.path.join(root, "attention"))
    file_list = os.listdir(os.path.join(root, "attention"))
    for file_idx, file_name in enumerate(file_list):
        for i in range(CONF.TRAIN.MAX_LENGTH):
            if os.path.exists(os.path.join(root, "attention", file_name, "{}.png".format(i))):
                print("rendered and skipped: {}/{}.png".format(file_name, i))
            else:
                print("rendering: {}/{}.nrrd, {} models left".format(file_name, i, len(file_list) - file_idx - 1))
                _ = subprocess.call(
                    ["ssc/render-voxels.js", "--input", os.path.join(root, "attention", "{}/{}.nrrd".format(file_name, i))], 
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL
                )

def main(args):
    root = os.path.join("../korhal-home/3d_captioning", CONF.PATH.OUTPUT_EMBEDDING, args.path)
    # render nrrd
    print("start rendering...\n")
    render(root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the pretrained encoders")
    args = parser.parse_args()
    main(args)