import numpy as np
import os
import pandas
import argparse
import time
import math
import h5py
import json
from PIL import Image

def main(args):
    verbose = args.verbose
    coco_root = "/mnt/raid/davech2y/COCO_2014/"
    coco_size = 64
    for phase in ["train", "val"]:
        print("phase: ", phase)
        # settings
        coco_dir = os.path.join(coco_root, "%s2014" % phase)
        coco_cap = os.path.join(coco_root, "annotations", "captions_%s2014.json" % phase)
        coco_paths = None
        database = h5py.File(os.path.join(coco_root, "preprocessed", "coco_%s2014.hdf5" % phase), "w")  

        # processing captions
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
            coco_paths = coco_csv.file_name.values.tolist()

        # processing images
        dataset = database.create_dataset("images", (len(coco_paths), 3 * coco_size * coco_size), dtype="float")
        for i, path in enumerate(coco_paths):
            start_since = time.time()
            image = np.array(Image.open(os.path.join(coco_dir, path)).resize((coco_size, coco_size)))
            if len(image.shape) < 3:
                temp = np.zeros((coco_size, coco_size, 3))
                temp[:, :, 0] = temp[:, :, 1] = temp[:, :, 2] = image
                image = np.reshape(temp, (-1))
            else:
                image = np.reshape(image[:, :, :3], (-1))
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            dataset[i] = image
            exetime_s = time.time() - start_since
            eta_s = exetime_s * (len(coco_paths) - i)
            eta_m = math.floor(eta_s / 60)
            if i % verbose == 0: 
                print("preprocessed and stored: %d, ETA: %dm %ds" % (i, eta_m, eta_s - eta_m * 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    args = parser.parse_args()
    print(args)
    print()
    main(args)