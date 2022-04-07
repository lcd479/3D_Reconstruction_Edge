# --- 기본 ---- 
import numpy as np
import os
import time
from IPython.display import clear_output
from config import cfg, update_config
import matplotlib.pyplot as plt
import warnings
import argparse
import datetime

# ---- torch ---
import torch
import torch.distributed as dist
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.data as torchdata

# ---- loger ----
from tensorboardX import SummaryWriter
from loguru import logger

# ---- utils & dataset & network
from datasets.sampler import DistributedSampler
from datasets import transforms, find_dataset_def
from utils import *
from datasets.scannet import *
from network.NeuralRecon import *
from ops.comm import *

# ---- argparse ----

def args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of NeuralRecon')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    # parse arguments and check
    args = parser.parse_args()

    return args

# ---- device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tb_writer = SummaryWriter('./runs/03.28/')
# ---- Read cfg ----

args = args()
update_config(cfg, args)
cfg.defrost()
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
print('number of gpus: {}'.format(num_gpus))
cfg.DISTRIBUTED = num_gpus > 1
cfg.LOCAL_RANK = args.local_rank
cfg.freeze()

torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# ---- create logger ----
if not os.path.isdir(cfg.LOGDIR):
    os.makedirs(cfg.LOGDIR)
    
current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
logfile_path = os.path.join(cfg.LOGDIR, f'{current_time_str}_{cfg.MODE}.log')
print('creating log file', logfile_path)
logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

# ---- parameters ---- 
n_views = 9
random_rotation = cfg.TRAIN.RANDOM_ROTATION_3D
random_translation = cfg.TRAIN.RANDOM_TRANSLATION_3D
paddingXY = cfg.TRAIN.PAD_XY_3D
paddingZ = cfg.TRAIN.PAD_Z_3D

# ---- transforms ----
transform = []
transform += [transforms.ResizeImage((640,480)),
              transforms.ToTensor(),
              transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                  paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
              transforms.IntrinsicsPoseToProjection(n_views, 4)]             
transforms = transforms.Compose(transform)

# --- Load Dataset ----

path = './datasets/'
MVSDdataset = find_dataset_def('scannet')
train_dataset = MVSDdataset(datapath = cfg.TRAIN.PATH, mode = "train" , transforms = transforms, n_views = cfg.TRAIN.N_VIEWS, n_scales = len(cfg.MODEL.THRESHOLDS) - 1)
TrainImgLoader = DataLoader(train_dataset, batch_size = cfg.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

# ---- load Model & Optimizer ----

model = NeuralRecon(cfg)
model = torch.nn.DataParallel(model, device_ids=[0])
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD)


def train_sample(sample):
    model.train()
    optimizer.zero_grad()

    outputs, loss_dict = model(sample)
    loss = loss_dict['total_loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return tensor2float(loss), tensor2float(loss_dict)

# ---- Train ----
start_epoch = 0

if cfg.RESUME:
    saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key = lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if len(saved_models) != 0 :
       
        loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
        logger.info("resuming " + str(loadckpt))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
        state_dict = torch.load(loadckpt, map_location=map_location)
        model.load_state_dict(state_dict['model'], strict=False)
        optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        start_epoch = state_dict['epoch'] + 1
        
    elif cfg.LOADCKPT != '':
        # load checkpoint file specified by args.loadckpt
        logger.info("loading model {}".format(cfg.LOADCKPT))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
        state_dict = torch.load(cfg.LOADCKPT, map_location=map_location)
        model.load_state_dict(state_dict['model'])
        optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        start_epoch = state_dict['epoch'] + 1
    
logger.info("start at epoch {}".format(start_epoch))
logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

milestones = [int(epoch_idx) for epoch_idx in cfg.TRAIN.LREPOCHS.split(':')[0].split(',')]
lr_gamma = 1 / float(cfg.TRAIN.LREPOCHS.split(':')[1])
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,last_epoch=start_epoch - 1)

for epoch_idx in range(start_epoch ,cfg.TRAIN.EPOCHS):    
    
    logger.info('Epoch {}:'.format(epoch_idx))
    lr_scheduler.step()  
    
    TrainImgLoader.dataset.epoch = epoch_idx
    TrainImgLoader.dataset.tsdf_cashe = {}

    for batch_idx, sample in enumerate(TrainImgLoader):
        global_step = len(TrainImgLoader) * epoch_idx + batch_idx
        do_summary = global_step & cfg.SUMMARY_FREQ == 0
        start_time = time.time()
        
        loss, scalar_outputs = train_sample(sample)
        logger.info('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, LR {}, time = {:.3f}'.format(epoch_idx,
                                                                                         cfg.TRAIN.EPOCHS,
                                                                                         batch_idx,
                                                                                         len(TrainImgLoader),
                                                                                         loss,
                                                                                         optimizer.param_groups[0]['lr'],  
                                                                                         time.time() - start_time))

        if do_summary:
            save_scalars(tb_writer, 'train', scalar_outputs, global_step)
        del scalar_outputs
        
        

    if (epoch_idx + 1) %1 == 0 :
        torch.save({
            'epoch': epoch_idx,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            "{}/model_{:0>6}.ckpt".format(cfg.LOGDIR, epoch_idx+1))

    

        

