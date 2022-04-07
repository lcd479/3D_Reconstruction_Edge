from .SwinTransformer import *
from .Swin_utils import *

import torch
import torch.nn.functional as F
import torch.nn as nn

class backbone(nn.Module):
    
    def __init__(self, img_size = (640,480)):
        super(backbone, self).__init__()
        
        self.img_size = img_size
        self.Backbone_Swin = Backbone_Swin(img_size= (640,480), embed_dim=128,depths=[2,2,18,2], 
                                 num_heads=[4,8,16,32], drop_path_rate=0.2, window_size=12,use_checkpoint=True)
        self.Backbone_Swin.load_from('./swin_base_patch4_window12_384_22k.pth')        
        
        in_f = [1024, 512, 256,128]
        out_f = [80, 40,24]

        self.inner0 = nn.Conv2d(out_f[1], out_f[0], 1, bias = True)
        self.inner1 = nn.Conv2d(out_f[2], out_f[0], 1, bias = True)

        self.conv0 = nn.Conv2d(in_f[3], out_f[2], 1, bias = False)
        self.conv1 = nn.Conv2d(in_f[2], out_f[1], 1, bias = False)
        self.conv2 = nn.Conv2d(in_f[1], out_f[0], 1, bias = False)
        self.conv3 = nn.Conv2d(in_f[0], out_f[0], 1, bias = False)
        
        self.out0 = nn.Conv2d(out_f[0], out_f[0], 1, bias = False)
        self.out1 = nn.Conv2d(out_f[0], out_f[1], 1, bias = False)
        self.out2 = nn.Conv2d(out_f[0], out_f[2], 1, bias = False)
        
        self.UpC = nn.Conv2d(1, 3, 1, bias = False)
        
    def forward(self, rgb, edge): 
        
        rgb_out_list = self.Backbone_Swin(rgb)
        edge_out_list = self.Backbone_Swin(self.UpC(edge))
        
        out_list = rgb_out_list+ edge_out_list
        
        T_out0 = self.conv0(out_list[0])
        T_out1 = self.conv1(out_list[1])
        T_out2 = self.conv2(out_list[2])
        
        outputs = []
        intra_feat = T_out2
        out = self.out0(intra_feat)
        outputs.append(out)
     
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode='nearest') + self.inner0(T_out1)
        out = self.out1(intra_feat)
        outputs.append(out)
        
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode='nearest') + self.inner1(T_out0)
        out = self.out2(intra_feat)
        outputs.append(out)


        return outputs[::-1]
