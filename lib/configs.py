from easydict import EasyDict

CONF = EasyDict()


'''
global configurations for path
'''

CONF.PATH = EasyDict()
# general
CONF.PATH.DATA_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/" # TODO mofidy this
# CONF.PATH.SHAPENET_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_{}_solid" # TODO mofidy this
CONF.PATH.SHAPENET_ROOT = "data/nrrd_256_filter_div_{}_solid" # TODO mofidy this
CONF.PATH.SHAPENET_SPLIT_ROOT = "/home/davech2y/text2shape/pretrained/shapenet/" # TODO modify this
CONF.PATH.SHAPENET_DATABASE = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_{}_solid.hdf5" # TODO modify this
CONF.PATH.SHAPENET_IMG = "{}/{}.png" # model_id
CONF.PATH.SHAPENET_NRRD = "{}/{}.nrrd" # model_id
CONF.PATH.SHAPENET_PROBLEMATIC = "data/shapenet_problematic.p"
CONF.PATH.PRIMITIVES_ROOT = "data/primitives_{}" # resolution
CONF.PATH.PRIMITIVES_IMG = "{}/{}.png" # cat_id, model_id
CONF.PATH.PRIMITIVES_NRRD = "{}/{}.nrrd" # cat_id, model_id
# path to split
CONF.PATH.SPLIT_NAME = "processed_captions_{}.p"
# output path
CONF.PATH.OUTPUT_EMBEDDING = "outputs/embedding/"
CONF.PATH.OUTPUT_CAPTION = "outputs/caption/"
# path to pretrained embeddings
CONF.PATH.SHAPENET_PRETRAINED = "pretrained/shapenet_embeddings_{}.p" # split
CONF.PATH.PRIMITIVE_PRETRAINED = "pretrained/primitive_embeddings_{}.p" # split
# path to trained embeddings
CONF.PATH.SHAPENET_EMBEDDING = "outputs/embedding/shapenet_embeddings_{}.p" # split
CONF.PATH.PRIMITIVE_EMBEDDING = "outputs/embedding/primitive_embeddings_{}.p" # split


'''
global configurations for training
'''

CONF.TRAIN = EasyDict()
# parameters of training
CONF.TRAIN.SETTINGS = "{}_v{}_trs{}_lr{}_wd{}_e{}_bs{}_{}" 
# mode, voxel, train_size, learning_rate, weight_decay, batch_size, num_process, attention_type 
CONF.TRAIN.DATASET = 'shapenet'
CONF.TRAIN.PRIMITIVES_NUM_PER_MODEL = 10
CONF.TRAIN.N_CAPTION_PER_MODEL = 2
CONF.TRAIN.RANDOM_SAMPLE = False
CONF.TRAIN.REDUCE_STEP = 10
CONF.TRAIN.REDUCE_FACTOR = 0.95
CONF.TRAIN.CLIP_VALUE = 5.
CONF.TRAIN.N_NEIGHBOR = 10
CONF.TRAIN.EVAL_DATASET = 'val'
CONF.TRAIN.EVAL_MODE = 't2s'
CONF.TRAIN.EVAL_METRIC = 'cosine'
# max length of captions
CONF.TRAIN.MAX_LENGTH = 20
# self attention
CONF.TRAIN.ATTN = "noattention"
# hyperparamters
CONF.TRAIN.RESOLUTION = 32
CONF.TRAIN.TRAIN_SIZE = -1
CONF.TRAIN.VAL_SIZE = -1
CONF.TRAIN.LEARNING_RATE = 2e-4
CONF.TRAIN.WEIGHT_DECAY = 5e-4
CONF.TRAIN.VERBOSE = 10
# fixed batch size
if CONF.TRAIN.DATASET == 'shapenet':
    CONF.TRAIN.BATCH_SIZE = 100
elif CONF.TRAIN.DATASET == 'primitives':
    CONF.TRAIN.BATCH_SIZE = 75
else:
    raise ValueError("invalid dataset, terminating...")



'''
global configurations for LBA
'''

CONF.LBA = EasyDict()
# flags of LBA
CONF.LBA.IS_LBA = True
CONF.LBA.IS_LBA_TST = True
CONF.LBA.IS_LBA_STS = True
CONF.LBA.IS_LBA_VISIT = True
# weights
CONF.LBA.WALKER_WEIGHT = 1.
CONF.LBA.VISIT_WEIGHT = 0.25


'''
global configurations for ML
'''

CONF.ML = EasyDict()
# flags of ML
CONF.ML.IS_ML = True
CONF.ML.IS_ML_TT = True
CONF.ML.IS_ML_SS = True
CONF.ML.IS_ML_ST = True
CONF.ML.COSINE_DISTANCE = True
CONF.ML.INVERTED_LOSS = True
# parameters
CONF.ML.METRIC_MULTIPLIER = 1.
CONF.ML.METRIC_MARGIN = 1.0


'''
global configurations for normalization
'''

CONF.NORM = EasyDict()
# flags of normalization
CONF.NORM.IS_NORM_PENALTY = True
# paramters
CONF.NORM.MAX_NORM = 10.
CONF.NORM.TEXT_NORM_MULTIPLIER = 2.
CONF.NORM.SHAPE_NORM_MULTIPLIER = 2.


'''
global configurations for multi-head training
'''
CONF.MH = EasyDict()
CONF.MH.ATTN_MULTIPLIER = 0.5
