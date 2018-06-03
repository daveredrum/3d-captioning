import numpy as np
import os
import pandas
import argparse
import time
import math
import h5py
import json
import torchvision.transforms as transforms
from PIL import Image

def main(args):
    verbose = args.verbose
    phases = [args.phases]
    if not phases:
        phases = ["train", "valid"]
    coco_size = args.size
    coco_root = "/mnt/raid/davech2y/COCO_2014/"
    for phase in phases:
        print("phase: ", phase)
        # settings
        coco_dir = os.path.join(coco_root, "%s2014" % phase)
        coco_cap = os.path.join(coco_root, "annotations", "captions_%s2014.json" % phase)
        coco_paths = None
        database = h5py.File(os.path.join(coco_root, "preprocessed", "coco_%s2014_%d.hdf5" % (phase, coco_size)), "w", libver='latest')  

        # processing captions
        print("creating preprocessed csv...")
        print()
        with open(coco_cap) as f:
            cap_json = json.load(f)
            df_caption = {
                'image_id': [item["image_id"] for item in cap_json["annotations"]],
                'caption': [item["caption"] for item in cap_json["annotations"]]
            }
            df_caption = pandas.DataFrame(df_caption, columns=['image_id', 'caption'])
            df_filename = {
                'image_id': [item["id"] for item in cap_json["images"]],
                'file_name': [item["file_name"] for item in cap_json["images"]]
            }
            df_filename = pandas.DataFrame(df_filename, columns=['image_id', 'file_name'])
            coco_csv = df_caption.merge(df_filename, how="inner", left_on="image_id", right_on="image_id")
            coco_csv = coco_csv.sample(frac=1).reset_index(drop=True)
            # shuffle the dataset
            coco_csv.to_csv(os.path.join(coco_root, "preprocessed", "coco_%s2014.caption.csv" % phase), index=False)
            coco_paths = coco_csv.file_name.drop_duplicates().values.tolist()
            # create indices
            print("creating indices for images...")
            print()
            index = {coco_paths[i]: i for i in range(len(coco_paths))}
            __all = coco_csv.file_name.values.tolist()
            mapping = {i: index[__all[i]] for i in range(len(__all))}
            print("saving indices...")
            print()
            with open(os.path.join(coco_root, "preprocessed", "%s_index.json" % phase)) as f:
                json.dump(mapping, f)


        # processing images
        print("preprocessing the images...")
        print()
        dataset = database.create_dataset("images", (len(coco_paths), 3 * coco_size * coco_size), dtype="float")
        trans = transforms.Compose([
            transforms.Resize(coco_size),
            transforms.CenterCrop(coco_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for i, path in enumerate(coco_paths):
            start_since = time.time()
            if not index[path]:
                image = Image.open(os.path.join(coco_dir, path)).convert('RGB')
                image = trans(image)
                image = image.view(-1).numpy()
                dataset[i] = image
                index[path] = i
            else:
                image = dataset[index[path]]
                dataset[i] = image
            exetime_s = time.time() - start_since
            eta_s = exetime_s * (len(coco_paths) - i)
            eta_m = math.floor(eta_s / 60)
            if i % verbose == 0: 
                print("preprocessed and stored: %d, ETA: %dm %ds" % (i, eta_m, eta_s - eta_m * 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--phases", type=str, default=None, help="train/valid")
    parser.add_argument("--size", type=int, default=64, help="height/width of the images")
    args = parser.parse_args()
    print(args)
    print()
    main(args)