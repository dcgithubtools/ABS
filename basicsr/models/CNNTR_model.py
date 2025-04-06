import torch
from torch.nn import functional as F

from collections import OrderedDict
from utils.registry import MODEL_REGISTRY
from models.sr_model import SRModel
from thop import profile
import math
from tqdm import tqdm
from os import path as osp

@MODEL_REGISTRY.register()
class CNNTRmodel(SRModel):

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

    def optimize_parameters(self, current_iter):

        # input = torch.randn(1, 3, 32, 32).to(self.device)
        # flops, params = profile(self.net_g, input)
        # print('FLOPs = ' + str(flops/1000**3) + 'G')

        self.optimizer_g.zero_grad()
        self.optimizer_g.zero_grad()
        # self.output, loss_ratio, cnn_f, sa_f, cnn_sr, sa_sr = self.net_g(self.lq)
        
        self.output, loss_ratio = self.net_g(self.lq)

        # contrastive loss
        # bic_sample = self.upsampler(self.lq)
        # cos_loss = self.cos_los(sa_sr, cnn_sr, bic_sample)

        # knowledge distll loss
        # dis_loss = 0
        # for i in range(len(cnn_f)):
        #     # cnn_f 16,60,64,64
        #     dis_loss += self.disloss(cnn_f[i], sa_f[i].detach())
        # dis_loss = dis_loss / 1000.0
        
        loss_ratio = loss_ratio.mean()

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss

        l_total += loss_ratio
        loss_dict['l_ratio'] = loss_ratio

        # l_total += cos_loss
        # loss_dict['cos_loss'] = cos_loss
        # l_total += dis_loss
        # loss_dict['dis_loss'] = dis_loss

        # if self.cri_perceptual:
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)