import torch.nn as nn
import torch.nn.functional as F
import archs.arch_util as arch_util
import torch
from utils.registry import ARCH_REGISTRY
from archs.FSRCNN_arch import FSRCNN

@ARCH_REGISTRY.register()
class FSRCNNR(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(FSRCNNR, self).__init__()
        self.upscale=4

        self.net1 = FSRCNN(in_nc,self.upscale,16,12,4)
        self.net2 = FSRCNN(in_nc,self.upscale,36,12,4)
        self.net3 = FSRCNN(in_nc,self.upscale,56,12,4)

        self.threshold1 = 0.82
        self.threshold2 = 0.94


        self.regressor = nn.Sequential(
            nn.Conv2d(3,64,3,3),
            
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),

            nn.Linear(64,2),
      
            nn.Tanh()
        )

    def forward(self, x, is_train):
        if is_train:
            reg_f = self.regressor(x)

            out1 = self.net1(x)
            out2 = self.net2(x)
            out3 = self.net3(x)

            return reg_f, out1, out2, out3
        
        else:

            reg_1 = self.regressor(x)
         


            reg_1_1 = torch.where(*reg_1[:,0]>self.threshold1,1,0)
            reg_1_2 = torch.where(*reg_1[:,1]>self.threshold2,1,0)

            if reg_1_1 == 0:
                flag = 0

                out = self.net1(x)
            elif reg_1_1 == 1 and reg_1_2 == 0:
                flag = 1

                out = self.net2(x)
            elif reg_1_1 == 1 and reg_1_2 == 1:
                flag = 2

                out = self.net3(x)

            
        return out, flag
