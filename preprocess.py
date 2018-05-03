import numpy as np
import os
import nrrd
import h5py
import pandas
import argparse

def main(args):
    shape_size = args.size
    verbose = args.verbose
    database_root = "/mnt/raid/davech2y/ShapeNetCore_vol/"
    database_name = "nrrd_256_filter_div_%s_solid.hdf5" % shape_size
    database = h5py.File(os.path.join(database_root, database_name), "w")

    shapenet_root = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_%s_solid/" % shape_size
    if not os.path.exists(shapenet_root):
        print("dataset does not exist, terminating...")
        return
    
    shapenet_csv = pandas.read_csv("/home/davech2y/3d_captioning/captions.tablechair.csv")
    shapenet_shapepath = [
        os.path.join(shapenet_root, model_name, model_name + '.nrrd') for model_name in shapenet_csv.modelId.values.tolist()
    ]
    shapenet_size = len(shapenet_shapepath)
    shapenet_shape = [3, shape_size, shape_size, shape_size]
    shapenet_shape_flat = np.prod(shapenet_shape)
    
    shapenet_dataset = database.create_dataset("shapes", (shapenet_size, shapenet_shape_flat), dtype="float")

    for i, shapepath in enumerate(shapenet_shapepath):
        if i % verbose == 0: 
            print("preprocessed and stored: ", i)
        shape, _ = nrrd.read(shapepath)
        shape = np.reshape(shape[:3], (-1))
        shape = (shape - np.min(shape)) / (np.max(shape) - np.min(shape))
        shapenet_dataset[i] = shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32, help="size of shapes")
    parser.add_argument("--verbose", type=int, default=1, help="show report")
    args = parser.parse_args()
    print(args)
    print()
    main(args)