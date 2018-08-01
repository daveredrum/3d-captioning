
# path for shapenet
VOXEL = 32
DATA_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/" # TODO mofidy this
SHAPE_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_{}_solid" # TODO mofidy this
PRIMITIVE_ROOT = "/mnt/raid/davech2y/text2shape_primitives"

# nrrd
SHAPENET_NRRD = "{}/{}.nrrd" # model_id

# path to split
SPLIT_ROOT = "/home/davech2y/text2shape/pretrained/shapenet/" # TODO modify this
SPLIT_NAME = "processed_captions_{}.p"

# path to pretrained embeddings
SHAPENET_PRETRAINED = "pretrained/shapenet_embeddings_{}.p" # split
PRIMITIVE_PRETRAINED = "pretrained/primitive_embeddings_{}.p" # split

# path to trained embeddings
SHAPENET_EMBEDDING = "outputs/embedding/shapenet_embeddings_{}.p" # split
PRIMITIVE_EMBEDDING = "outputs/embedding/primitive_embeddings_{}.p" # split

# max length of captions
MAX_LENGTH = 18

# parameters of training
REDUCE_STEP = 10
WALKER_WEIGHT = 1.
VISIT_WEIGHT = 0.25
METRIC_MULTIPLIER = 1.
METRIC_MARGIN = 0.5
CLIP_VALUE = 5.

# parameters of metric learning
COSINE_DISTANCE = True
INVERTED_LOSS = True