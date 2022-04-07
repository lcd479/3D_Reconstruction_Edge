from yacs.config import CfgNode as CN

_C = CN()


_C.MODE = 'train'
_C.DATASET = 'scannet'
_C.BATCH_SIZE = 1
_C.LOADCKPT = ''
_C.LOGDIR = './checkpoints/'
_C.RESUME = True
_C.SUMMARY_FREQ = 20
_C.SAVE_FREQ = 1
_C.SEED = 1
_C.SAVE_SCENE_MESH = False
_C.SAVE_INCREMENTAL = False
_C.VIS_INCREMENTAL = False
_C.REDUCE_GPU_MEM = False

_C.LOCAL_RANK = 0
_C.DISTRIBUTED = False


# train
_C.TRAIN = CN()
_C.TRAIN.PATH = ''
_C.TRAIN.EPOCHS = 20
_C.TRAIN.LR = 1e-3
_C.TRAIN.LREPOCHS = '12,24,48:2'
_C.TRAIN.WD = 0.0
_C.TRAIN.N_VIEWS = 9
_C.TRAIN.N_WORKERS = 64
_C.TRAIN.RANDOM_ROTATION_3D = True
_C.TRAIN.RANDOM_TRANSLATION_3D = True
_C.TRAIN.PAD_XY_3D = .1
_C.TRAIN.PAD_Z_3D = .025

# model
_C.MODEL = CN()
_C.MODEL.N_VOX = [96, 96, 96]
_C.MODEL.VOXEL_SIZE = 0.04
_C.MODEL.THRESHOLDS = [0, 0, 0]
_C.MODEL.N_LAYER = 3

_C.MODEL.TRAIN_NUM_SAMPLE = [4096, 16384, 65536]
_C.MODEL.TEST_NUM_SAMPLE = [4096, 16384, 65536]

_C.MODEL.LW = [1.0, 0.8, 0.64]

# TODO: images are currently loaded RGB, but the pretrained models expect BGR
_C.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_STD = [1., 1., 1.]
_C.MODEL.THRESHOLDS = [0, 0, 0]
_C.MODEL.POS_WEIGHT = 1.5

_C.MODEL.BACKBONE2D = CN()
_C.MODEL.BACKBONE2D.ARC = 'fpn-mnas-1'

_C.MODEL.SPARSEREG = CN()
_C.MODEL.SPARSEREG.DROPOUT = False

_C.MODEL.FUSION = CN()
_C.MODEL.FUSION.FUSION_ON = False
_C.MODEL.FUSION.HIDDEN_DIM = 64
_C.MODEL.FUSION.AVERAGE = False
_C.MODEL.FUSION.FULL = True


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


def check_config(cfg):
    pass
