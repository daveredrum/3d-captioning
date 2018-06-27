
# path for shapenet
DATA_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/" # TODO mofidy this
SHAPE_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid" # TODO mofidy this
PRIMITIVE_ROOT = "/mnt/raid/davech2y/text2shape_primitives"

# path to split
SPLIT_ROOT = "/home/davech2y/text2shape/data/shapenet/" # TODO modify this
SPLIT_NAME = "processed_captions_{}.p"

# path to pretrained embeddings
SHAPENET_EMBEDDING = "data/shapenet_embeddings_{}.p"
PRIMITIVE_EMBEDDING = "data/primitive_embeddings_{}.p"

# max length of captions
MAX_LENGTH = 18