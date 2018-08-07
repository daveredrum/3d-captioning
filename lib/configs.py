
# path for shapenet
VOXEL = 32
DATA_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/" # TODO mofidy this
SHAPE_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_{}_solid" # TODO mofidy this
PRIMITIVE_ROOT = "/mnt/raid/davech2y/text2shape_primitives"

# nrrd
SHAPENET_NRRD = "{}/{}.nrrd" # model_id
PROBLEMATIC = "data/shapenet_problematic.p"

# path to split
SPLIT_ROOT = "/home/davech2y/text2shape/pretrained/shapenet/" # TODO modify this
SPLIT_NAME = "processed_captions_{}.p"

# output path
OUTPUT_EMBEDDING = "outputs/embedding/"
OUTPUT_CAPTION = "outputs/caption/"

# path to pretrained embeddings
SHAPENET_PRETRAINED = "pretrained/shapenet_embeddings_{}.p" # split
PRIMITIVE_PRETRAINED = "pretrained/primitive_embeddings_{}.p" # split

# path to trained embeddings
SHAPENET_EMBEDDING = "outputs/embedding/shapenet_embeddings_{}.p" # split
PRIMITIVE_EMBEDDING = "outputs/embedding/primitive_embeddings_{}.p" # split

# max length of captions
MAX_LENGTH = 18

# parameters of training
SETTINGS = "{}_v{}_trs{}_lr{}_wd{}_e{}_bs{}_mp{}" # mode, voxel, train_size, learning_rate, weight_decay, batch_size, num_process
N_CAPTION_PER_MODEL = 2
REDUCE_STEP = 10
REDUCE_FACTOR = 0.95
WALKER_WEIGHT = 1.
VISIT_WEIGHT = 0.25
CLIP_VALUE = 5.

# parameters of ML
COSINE_DISTANCE = True
INVERTED_LOSS = True
METRIC_MULTIPLIER = 1.
METRIC_MARGIN = 0.5
MAX_NORM = 10.
TEXT_NORM_MULTIPLIER = 2.
SHAPE_NORM_MULTIPLIER = 2.