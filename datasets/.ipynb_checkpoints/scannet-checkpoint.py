import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from PIL import Image


import torch
from torch.utils.data import Dataset

import kornia.color as kc
import kornia.filters as kf

class ScanNetDataset(Dataset):
    
    def __init__(self, datapath, mode, transforms, n_views, n_scales):
        super(ScanNetDataset, self).__init__()
        
        self.datapath = datapath
        self.mode = mode
        self.n_views = n_views
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)
        
        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'
                     
        self.n_scales = n_scales
        self.epoch = 0
        self.tsdf_cashe = {}
        self.max_cashe = 100
        
    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
            
        return metas
        
    def __len__(self):
        return len(self.metas)
    
    def read_cam_file(self, filepath, vid):
        
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter =' ')[:3,:3]
        intrinsics = intrinsics.astype(np.float32)
        
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        
        return intrinsics, extrinsics
    
    def read_img(self, filepath):

        img = Image.open(filepath)
        return img
    
    def read_depth(self, filepath):
        
        depth_img = cv2.imread(filepath, -1).astype(np.float32)
        depth_img = depth_img / 1000.
        depth_img[depth_img > 3.0] = 0
            
        return depth_img
    
    def make_edge(self, img_filepath):
        
        img = Image.open(img_filepath).convert('L')
        img_array = np.array(img)
        edge = cv2.resize(cv2.Canny(img_array, 80, 215), (640,480))
        return np.expand_dims(edge, axis= 0)     
    
    def read_scene_volumes(self, data_path, scene):
        
        
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
                
            full_tsdf_list = []
            for l in range(self.n_scales +1):
                # Laod full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
                
            self.tsdf_cashe[scene] = full_tsdf_list
            
        return self.tsdf_cashe[scene]
    
    def __getitem__(self, index):
        
        meta = self.metas[index]
        
        imgs = []
        depth =[]
        edge = []
        extrinsics_list = []
        intrinsics_list = []
        
        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])
        
        
        
        for i, vid in enumerate(meta['image_ids']):
            #Load image
            imgs.append(self.read_img(os.path.join(self.datapath, self.source_path, meta['scene'], 'color','{}.jpg'.format(vid))))
            depth.append(self.read_depth(os.path.join(self.datapath, self.source_path, meta['scene'], 'depth','{}.png'.format(vid))))
            edge.append(self.make_edge(os.path.join(self.datapath, self.source_path, meta['scene'], 'color','{}.jpg'.format(vid))))
            
            #Load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),vid)
            
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)
            
        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        
        items = {
            'imgs' : imgs,
            'depth' : depth,
            'edge' : edge,
            'intrinsics' : intrinsics,
            'extrinsics' : extrinsics,
            'tsdf_list_full' : tsdf_list,
            'vol_origin' : meta['vol_origin'],
            'scene' : meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch' : [self.epoch]
        }
        if self.transforms is not None :
            
            items = self.transforms(items)
            
        return items
    
    
import importlib

def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    if dataset_name == 'scannet':
        return getattr(module, "ScanNetDataset")
    elif dataset_name == 'demo':
        return getattr(module, "DemoDataset")
