import torch
from torch.nn import functional as F

from utils.registry import MODEL_REGISTRY
from models.sr_model import SRModel

import math
from tqdm import tqdm
from os import path as osp
from utils import imwrite,tensor2img
import time

from thop import profile

@MODEL_REGISTRY.register()
class TileModel(SRModel):

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def test(self):

        # input = torch.randn(1, 3, 64, 64).to(self.device)
        # flops, params = profile(self.net_g, inputs=(input,True))
        # print('FLOPs = ' + str(flops/1000**3) + 'G')
        # print(params)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                
                self.output = self.tile_test(self.lq, self.net_g_ema)
        else:
            self.net_g.eval()
            with torch.no_grad():

                # time_s = time.time()
                self.output, self.flag_1, self.flag_2, self.flag_3 = self.tile_test(self.lq, self.net_g)
                # time_e = time.time()
                # ti = time_e - time_s
                # self.t = self.t + ti
                # print(ti)
                # print(self.t/100.0)
            

            self.net_g.train()

    def tile_test(self, img_lq, model):
        tile = self.opt['tile']
        
        if tile == 0:
            # test the image as a whole
            output = model(img_lq)
        else:

            # test the image tile by tile
            b, c, h, w = img_lq.size()
            tile = min(tile, h, w)
            tile_overlap = tile//16
            sf =  self.opt['scale']

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            flag_list1 = [] 
            flag_list2 = [] 
            flag_list3 = [] 
            x=0

            self.per_i = 0
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    # out_patch, _, _, _, _ = model(in_patch, 1)
                    # out_patch = model(in_patch)

                    time_s = time.time()
                    out_patch, flag = model(in_patch, False)
                    time_e = time.time()
                    ti = time_e - time_s
                    print(ti)
                    self.per_i = self.per_i + ti
                    

                    # out_x = tensor2img(out_patch)

                    # x = x+1
                    # x_s = str(x)
                    # x_s = x_s+'.png'
                    # if flag == 1:
                    #     print(x)

                    # save_img_path = osp.join('/home/zhao/dc/codex/results/r/', x_s)
                    # imwrite(out_x, save_img_path)


                    out_patch_mask = torch.ones_like(out_patch)
                    
                    if flag == 0:
                        flag_list1.append(flag)
                    elif flag == 1:
                        flag_list2.append(flag)
                    elif flag == 2:
                        flag_list3.append(flag)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)

            print(self.per_i)
            
            # self.t = self.t + self.per_i
            # print(self.t/100.0)
            
            
            
            output = E.div_(W)

        return output, flag_list1, flag_list2, flag_list3