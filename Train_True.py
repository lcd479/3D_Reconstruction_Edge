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
tb_writer = SummaryWriter('./runs/03.20/')
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

# ---- Test Parmeters---
t_n_views = 9
t_random_rotation = False
t_random_translation = False
t_paddingXY = 0
t_paddingZ = 0

# ---- transforms ----
Train_transform = []
Train_transform += [transforms.ResizeImage((640,480)),
                    transforms.ToTensor(),
                    transforms.RandomTransformSpace(
                    cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                    paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
                    transforms.IntrinsicsPoseToProjection(n_views, 4)]             
Train_transform = transforms.Compose(Train_transform)


# ---- Test transforms ---

Test_transform = []
Test_transform += [transforms.ResizeImage((640,480)),
                  transforms.ToTensor(),
                  transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, t_random_rotation, t_random_translation,
                  t_paddingXY, t_paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
                  transforms.IntrinsicsPoseToProjection(n_views, 4)]             
Test_transform = transforms.Compose(Test_transform)

# --- Load Dataset ----

path = './datasets/'
MVSDdataset = find_dataset_def('scannet')
train_dataset = MVSDdataset(datapath = cfg.TRAIN.PATH, mode = "train" , transforms = Train_transform, n_views = cfg.TRAIN.N_VIEWS, n_scales = len(cfg.MODEL.THRESHOLDS) - 1)
TrainImgLoader = DataLoader(train_dataset, batch_size = cfg.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

test_dataset = MVSDdataset(datapath = cfg.TEST.PATH, mode = "test" , transforms = Test_transform, n_views = cfg.TEST.N_VIEWS, n_scales = len(cfg.MODEL.THRESHOLDS) - 1)
TestImgLoader = DataLoader(test_dataset, batch_size = cfg.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

# ---- load Model & Optimizer ----

model = NeuralRecon(cfg)
model = torch.nn.DataParallel(model, device_ids=[0])
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD)



# ---- Test ----

def test_sample(sample, save_scene=False):
    model.eval()

    outputs, loss_dict = model(sample, save_scene)
    loss = loss_dict['total_loss']

    return tensor2float(loss), tensor2float(loss_dict), outputs


# ---- Train ----

def train_sample(sample):
    model.train()
    optimizer.zero_grad()

    outputs, loss_dict = model(sample)
    loss = loss_dict['total_loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return tensor2float(loss), tensor2float(loss_dict)

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
        
        
    if (epoch_idx + 1) %1 == 0 :      
        ckpt_list = []
   
        #saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        #saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        #saved_models = saved_models[-1:]
        #for ckpt in saved_models:
            #if ckpt not in ckpt_list:
                # use the latest checkpoint file
                #loadckpt = os.path.join(cfg.LOGDIR, ckpt)
                #logger.info("resuming " + str(loadckpt))
                #state_dict = torch.load(loadckpt)
                #model.load_state_dict(state_dict['model'])
                #epoch_idx = state_dict['epoch']
        
        TestImgLoader.dataset.tsdf_cashe = {}
        
        avg_test_scalars = DictAverageMeter()
        save_mesh_scene = SaveScene(cfg)
        batch_len = len(TestImgLoader)
        for batch_idx, sample in enumerate(TestImgLoader):
               
            for n in sample['fragment']:
                logger.info(n)
                    # save mesh if SAVE_SCENE_MESH and is the last fragment
            save_scene = cfg.SAVE_SCENE_MESH and batch_idx == batch_len - 1
        
            start_time = time.time()
            loss, scalar_outputs, outputs = test_sample(sample, save_scene)
            logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                                    len(TestImgLoader),
                                                                                                    loss,
                                                                                                    time.time() - start_time))
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
        
            if batch_idx % 100 == 0:
                logger.info("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader),
                                                                               avg_test_scalars.mean()))
        
                        # save mesh
            if cfg.SAVE_SCENE_MESH:
                save_mesh_scene(outputs, sample, epoch_idx)
            save_scalars(tb_writer, 'fulltest', avg_test_scalars.mean(), epoch_idx)
            logger.info("epoch {} avg_test_scalars:".format(epoch_idx), avg_test_scalars.mean())
        
            #        ckpt_list.append(ckpt)
        

            


    

        

