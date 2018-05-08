import numpy as np
import os
import nrrd
import h5py
import pandas
import argparse
import time
import math
from PIL import Image

def main(args):
    visual_size = args.size
    image_size = 64
    verbose = args.verbose
    mode = args.mode
    database_root = "/mnt/raid/davech2y/ShapeNetCore_vol/"
    if mode == "2d":
        database_name = "nrrd_256_filter_div_%s_solid.png.hdf5" % visual_size
        database = h5py.File(os.path.join(database_root, database_name), "w")
        shapenet_root = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_%s_solid/" % visual_size
        if not os.path.exists(shapenet_root):
            print("dataset does not exist, terminating...")
            return
    
        shapenet_csv = pandas.read_csv("/home/davech2y/3d_captioning/captions.tablechair.csv")
        shapenet_shapepath = [
            os.path.join(shapenet_root, model_name, model_name + '.png') for model_name in shapenet_csv.modelId.values.tolist()
        ]
        shapenet_size = len(shapenet_shapepath)
        shapenet_image = [3, image_size, image_size]
        shapenet_image_flat = np.prod(shapenet_image)
        
        shapenet_dataset = database.create_dataset("images", (shapenet_size, shapenet_image_flat), dtype="float")

        for i, shapepath in enumerate(shapenet_shapepath):
            start_since = time.time()
            image = np.array(Image.open(shapepath).resize((image_size, image_size)))
            image = np.reshape(image[:, :, :3], (-1))
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            shapenet_dataset[i] = image
            exetime_s = time.time() - start_since
            eta_s = exetime_s * (len(shapenet_shapepath) - i)
            eta_m = math.floor(eta_s / 60)
            if i % verbose == 0: 
                print("preprocessed and stored: %d, ETA: %dm %ds" % (i, eta_m, eta_s - eta_m * 60))

    elif mode == "3d":
        database_name = "nrrd_256_filter_div_%s_solid.hdf5" % visual_size
        database = h5py.File(os.path.join(database_root, database_name), "w")
        shapenet_root = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_%s_solid/" % visual_size
        if not os.path.exists(shapenet_root):
            print("dataset does not exist, terminating...")
            return
    
        shapenet_csv = pandas.read_csv("/home/davech2y/3d_captioning/captions.tablechair.csv")
        shapenet_shapepath = [
            os.path.join(shapenet_root, model_name, model_name + '.nrrd') for model_name in shapenet_csv.modelId.values.tolist()
        ]
        shapenet_size = len(shapenet_shapepath)
        shapenet_shape = [3, visual_size, visual_size, visual_size]
        shapenet_shape_flat = np.prod(shapenet_shape)
        
        shapenet_dataset = database.create_dataset("shapes", (shapenet_size, shapenet_shape_flat), dtype="float")

        for i, shapepath in enumerate(shapenet_shapepath):
            start_since = time.time()
            shape, _ = nrrd.read(shapepath)
            shape = np.reshape(shape[:3], (-1))
            shape = (shape - np.min(shape)) / (np.max(shape) - np.min(shape))
            shapenet_dataset[i] = shape
            exetime_s = time.time() - start_since
            eta_s = exetime_s * (len(shapenet_shapepath) - i)
            eta_m = math.floor(eta_s / 60)
            if i % verbose == 0: 
                print("preprocessed and stored: %d, ETA: %dm %ds" % (i, eta_m, eta_s - eta_m * 60))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32, help="size of shapes")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    parser.add_argument("--mode", type=str, default="2d", help="specify the preprocessed data type")
    args = parser.parse_args()
    print(args)
    print()
    main(args)