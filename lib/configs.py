import os
from easydict import EasyDict

CONF = EasyDict()


'''
global configurations for path
'''

CONF.PATH = EasyDict()
# general
CONF.PATH.ROOT = "/home/davech2y/3d_captioning/" # TODO mofidy this
CONF.PATH.DATA_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/" # TODO mofidy this
CONF.PATH.PROC_DATA_ROOT = "/home/davech2y/ShapeNetCore_data" # TODO mofidy this
# CONF.PATH.SHAPENET_ROOT = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_{}_solid" # TODO mofidy this
CONF.PATH.SHAPENET_ROOT = "/home/davech2y/ShapeNetCore_data/nrrd_256_filter_div_{}_solid" # TODO mofidy this
CONF.PATH.SHAPENET_SPLIT_ROOT = "/home/davech2y/text2shape/pretrained/shapenet/" # TODO modify this
CONF.PATH.SHAPENET_DATABASE = "/mnt/raid/davech2y/ShapeNetCore_vol/nrrd_256_filter_div_{}_solid_3d.hdf5" # TODO modify this
CONF.PATH.SHAPENET_IMG = "{}/{}.png" # model_id
CONF.PATH.SHAPENET_NRRD = "{}/{}.nrrd" # model_id
CONF.PATH.SHAPENET_PROBLEMATIC = "/home/davech2y/ShapeNetCore_data/shapenet_problematic.p"
# CONF.PATH.PRIMITIVES_ROOT = "/home/davech2y/ShapeNetCore_data//primitives_{}" # TODO modify this
# CONF.PATH.PRIMITIVES_IMG = "{}/{}.png" # cat_id, model_id
# CONF.PATH.PRIMITIVES_NRRD = "{}/{}.nrrd" # cat_id, model_id
# path to split
CONF.PATH.SPLIT_NAME = "text2shape_split_{}_interm.p" # TODO mofidy this
# output path
CONF.PATH.OUTPUT_EMBEDDING = os.path.join(CONF.PATH.ROOT, "outputs/embedding/")
CONF.PATH.OUTPUT_CAPTION = os.path.join(CONF.PATH.ROOT, "outputs/caption/")
# for BERT
CONF.PATH.BERT_EMBEDDING = os.path.join(CONF.PATH.PROC_DATA_ROOT, "bert_embedding.hdf5")


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
# max length of captions
CONF.TRAIN.MAX_LENGTH = 20
# self attention
CONF.TRAIN.ATTN = "selfnew-sep-cf" # text2shape/self-nosep/self-sep/selfnew-nosep/selfnew-sep-p/selfnew-sep-sf/selfnew-sep-cf
# hyperparamters
CONF.TRAIN.RESOLUTION = 64
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
global configurations for BERT
'''

CONF.BERT = EasyDict()
# flags of BERT
CONF.BERT.IS_BERT = True
# configs of BERT
CONF.BERT.MODEL = "bert-large-uncased"
CONF.BERT.DIM = 1024

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
global configurations for captioning model
'''
CONF.CAP = EasyDict()
CONF.CAP.MAX_LENGTH = 18
CONF.CAP.DATASET = 'shapenet'
CONF.CAP.BEAM_SIZE = 1
CONF.CAP.VERBOSE = False
CONF.CAP.LEARNING_RATE = 1e-4
CONF.CAP.WEIGHT_DECAY = 2e-4
CONF.CAP.BATCH_SIZE = 100
CONF.CAP.ATTN = 'adaptive' # fc/att2in/att2all/adaptive
CONF.CAP.IS_EVAL = False
CONF.CAP.HIDDEN_SIZE = 512
CONF.CAP.EVAL_DATASET = 'val'  
CONF.CAP.SCHEDULE_THRESHOLD = 2.5
CONF.CAP.SAVE_THRESHOLD = 3.0

'''
global configurations for captioning model
'''
CONF.EVAL = EasyDict()
CONF.EVAL.RESOLUTION = 64
CONF.EVAL.ALPHA = 0.8
CONF.EVAL.EVAL_DATASET = 'val'
CONF.EVAL.EVAL_METRIC = 'cosine'
CONF.EVAL.NUM_TOP_K = 5
CONF.EVAL.NUM_CHOSEN = 500
CONF.EVAL.COMP_METHOD_A = "[FIN]shapenet_v64_trs11921_lr0.0002_wd0.0005_e20_bs100_text2shape"
CONF.EVAL.COMP_METHOD_B = "[FIN]shapenet_v64_trs11921_lr0.0002_wd0.0005_e20_bs100_self-nosep"
CONF.EVAL.COMP_CAP_A = {
    'FC': "[FIN]shapenet_text2shape_fc_trs59777_vs7435_e50_lr0.00010_w0.00001_bs256_vocab3521_beam1",
    'att2in': "[FIN]shapenet_text2shape_att2in_trs59777_vs7435_e50_lr0.00010_w0.00001_bs100_vocab3521_beam1",
    'att2all': "[FIN]shapenet_text2shape_att2all_trs59777_vs7435_e50_lr0.00010_w0.00020_bs100_vocab3521_beam1",
    'adaptive': "[FIN]shapenet_text2shape_adaptive_trs59777_vs7435_e50_lr0.00010_w0.00020_bs100_vocab3521_beam1"
}
CONF.EVAL.COMP_CAP_B = {
    'FC': "[FIN]shapenet_selfnew-sep-cf_fc_trs59777_vs7435_e50_lr0.00010_w0.00001_bs256_vocab3521_beam1",
    'att2in': "[FIN]shapenet_selfnew-sep-cf_att2in_trs59777_vs7435_e50_lr0.00010_w0.00001_bs100_vocab3521_beam1",
    'att2all': "[FIN]shapenet_selfnew-sep-cf_att2all_trs59777_vs7435_e50_lr0.00010_w0.00020_bs100_vocab3521_beam1",
    'adaptive': "[FIN]shapenet_selfnew-sep-cf_adaptive_trs59777_vs7435_e50_lr0.00010_w0.00020_bs100_vocab3521_beam1"
}