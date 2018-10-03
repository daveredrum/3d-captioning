import os
import argparse
import subprocess

# HACK
import sys
sys.path.append("korhal-home/3d_captioning") # you might want to change this ;)
from lib.configs import CONF

def filter_nrrd(file_list):
    filtered = []
    for file_full_name in file_list:
        file_name, file_ext = os.path.splitext(file_full_name)
        if file_ext == ".nrrd":
            filtered.append(file_name)
    
    return filtered

def render(root):
    dir_list = os.listdir(root)
    for dir_idx, dir_name in enumerate(dir_list):
        file_list = os.listdir(os.path.join(root, dir_name))
        for file_name in filter_nrrd(file_list):
            if os.path.exists(os.path.join(root, dir_name, "{}.png".format(file_name))):
                print("rendered and skipped: {}/{}.png".format(dir_name, file_name))
            else:
                print("rendering: {}/{}.nrrd, {} models left".format(dir_name, file_name, len(dir_list) - dir_idx - 1))
                _ = subprocess.call(
                    ["sstk/ssc/render-voxels.js", "--input", os.path.join(root, "{}/{}.nrrd".format(dir_name, file_name)), "--mesher", "greedy"], 
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL
                )


def main(args):
    root = args.path
    # render nrrd
    print("start rendering...\n")
    render(root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to the output root folder")
    args = parser.parse_args()
    main(args)