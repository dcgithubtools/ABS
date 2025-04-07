import torch.nn as nn
import torch.nn.functional as F
import archs.arch_util as arch_util
import torch
from utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class FSRCNNS(torch.nn.Module):
    def __init__(self, input_channels, scale, d, s, m, t):
        super(FSRCNNS, self).__init__()
        self.t = t
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []

        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
 

        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9,
                                            stride=scale, padding=3, output_padding=1)
        
        # t
        self.trans = nn.Conv2d(56, 56, 1, 1, 1)

        # s
        self.trans1 = nn.Sequential(
            nn.Conv2d(36, 56, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(56, 56, 3, 1, 1))
        
        self.trans2 = nn.Sequential(
            nn.Conv2d(36, 56, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(56, 56, 3, 1, 1))

        arch_util.default_init_weights([self.head_conv, self.body_conv, self.tail_conv, self.trans, self.trans1, self.trans2], 0.1)

    def forward(self, x):
        if self.t:
            feat_list = []

            fea1 = self.head_conv(x)
            fea1p = self.trans(fea1)
            feat_list.append(fea1p)

            # conv[x,y],左闭右开区间
            fea2 = self.body_conv(fea1)
            fea2p = self.trans(fea2)
            feat_list.append(fea2p)
            out = self.tail_conv(fea2)

            return out, feat_list
            
        else:      
            u_list = []
            b_list = []

            fea1 = self.head_conv(x)

            fea1u = self.trans1(fea1)
            u_list.append(fea1u)
            fea1b = self.trans2(fea1)
            b_list.append(fea1b)

            # conv[x,y],左闭右开区间
            fea2 = self.body_conv(fea1)
            
            fea2u = self.trans1(fea2)
            u_list.append(fea2u)
            fea2b = self.trans2(fea2)
            b_list.append(fea2b)

            out = self.tail_conv(fea2)
            
            return out, u_list, b_list