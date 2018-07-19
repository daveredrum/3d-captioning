
# path for shapenet
DATA_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/" # TODO mofidy this
SHAPE_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_128_solid" # TODO mofidy this
PRIMITIVE_ROOT = "/mnt/raid/davech2y/text2shape_primitives"

# nrrd
SHAPENET_NRRD = "{}/{}.nrrd" # model_id

# path to split
SPLIT_ROOT = "/home/davech2y/text2shape/pretrained/shapenet/" # TODO modify this
SPLIT_NAME = "processed_captions_{}.p"

# path to pretrained embeddings
SHAPENET_EMBEDDING = "pretrained/shapenet_embeddings_{}.p" # split
PRIMITIVE_EMBEDDING = "pretrained/primitive_embeddings_{}.p" # split

# max length of captions
MAX_LENGTH = 18

# parameters of joint encoding
WALKER_WEIGHT = 1.
VISIT_WEIGHT = 0.25
METRIC_MULTIPLIER = 1.
METRIC_MARGIN = 0.5
CLIP_VALUE = 5.