
# path for shapenet
DATA_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_32_solid" # TODO mofidy this

# path to split
SPLIT_ROOT = "/home/davech2y/text2shape/data/shapenet/" # TODO modify this
SPLIT_NAME = "processed_captions_{}.p"

# path to generated embeddings from test2shape
PRETRAINED_ROOT = "/home/davech2y/text2shape/outputs/shapenet/encoder_logdir/2018-06-19_12-31-47/test/"
PRETRAINED_TEXT_EMBEDDING = "text_embeddings_{}.p"
PRETRAINED_SHAPE_EMBEDDING = "shape_embeddings_{}.p"

# path to processed embeddings
PROCESSED_SHAPE_EMBEDDING = "data/shape_embeddings_{}.p"

# max length of captions
MAX_LENGTH = 18