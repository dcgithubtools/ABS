import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from copy import deepcopy

from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from metrics import calculate_psnr
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch.nn.functional as F

from thop import profile



@MODEL_REGISTRY.register()
class RGModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(RGModel, self).__init__(opt)
        # self.upsampler = torch.nn.Upsample(scale_factor=opt['network_g']['scale'], mode='bicubic')

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path_branch'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_path_branch1 = self.opt['path_branch'].get('branch1', None)
        load_path_branch2 = self.opt['path_branch'].get('branch2', None)
        load_path_branch3 = self.opt['path_branch'].get('branch3', None)

        if load_path_branch1 is not None:
            param_key = self.opt['path_branch'].get('param_key_g', 'params')
            self.load_n(self.net_g, load_path_branch1, load_path_branch2, load_path_branch3) 

        if self.is_train:
            self.init_training_settings()
    
    def load_n(self, network, load_path_branch1, load_path_branch2,  load_path_branch3):
        load_net1 = torch.load(load_path_branch1, map_location=lambda storage, loc: storage)
        load_net2 = torch.load(load_path_branch2, map_location=lambda storage, loc: storage)
        load_net3 = torch.load(load_path_branch3, map_location=lambda storage, loc: storage)
        param_key = 'params_ema'

        load_net1 = load_net1[param_key]
        load_net2 = load_net2[param_key]
        load_net3 = load_net3[param_key]

        network1 = network.net1
        network2 = network.net2
        network3 = network.net3

        for k, v in deepcopy(load_net1).items():
            if k.startswith('module.'):
                load_net1[k[7:]] = v
                load_net1.pop(k)
        network1.load_state_dict(load_net1, strict=True)

        for k, v in deepcopy(load_net2).items():
            if k.startswith('module.'):
                load_net2[k[7:]] = v
                load_net2.pop(k)
        network2.load_state_dict(load_net2, strict=True)

        for k, v in deepcopy(load_net3).items():
            if k.startswith('module.'):
                load_net3[k[7:]] = v
                load_net3.pop(k)
        network3.load_state_dict(load_net3, strict=True)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        
        if train_opt.get('cos_opt'):
            self.cos_los = build_loss(train_opt['cos_opt']).to(self.device)
        else:
            self.cos_los = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None   

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('grad_opt'):
            self.cri_grad = torch.nn.MSELoss().to(self.device)
        else:
            self.cri_grad = None 
        
        if train_opt.get('rf_opt'):
            self.cri_rf = build_loss(train_opt['rf_opt']).to(self.device)
        else:
            self.cri_rf = None 
        
        self.de_loss = torch.nn.MSELoss()

        # self.get_grad = Get_gradient()

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():  # can optimize for a part of the model
            if v.requires_grad and "regressor" not in k:
                v.requires_grad=False

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # print("flops")
        
        # input = torch.randn(1, 3, 32, 32).to(self.device)
        # flops, params = profile(self.net_g, inputs=input)
        # print('FLOPs = ' + str(flops/1000**3) + 'G')
        
        # print("ends")
        self.optimizer_g.zero_grad()

        self.reg, self.out1, self.out2, self.out3 = self.net_g(self.lq, True)

        # 32,3,256,256
        for i in range(len(self.out1)):
            sr_img1 = tensor2img(self.out1[i])
            sr_img2 = tensor2img(self.out2[i])
            sr_img3 = tensor2img(self.out3[i])
            self.img_gtx = tensor2img(self.gt[i])

            psnr1 = calculate_psnr(sr_img1, self.img_gtx, crop_border = 4, test_y_channel = True)
            psnr2 = calculate_psnr(sr_img2, self.img_gtx, crop_border = 4, test_y_channel = True)
            psnr3 = calculate_psnr(sr_img3, self.img_gtx, crop_border = 4, test_y_channel = True)
            
            psnrx = psnr2 - psnr1
            psnry = psnr3 - psnr2

            if i == 0:
                psnrx_res = torch.tensor(psnrx).unsqueeze(0)
                psnry_res = torch.tensor(psnry).unsqueeze(0)
            else:
                psnrx_res = torch.cat((psnrx_res, torch.tensor(psnrx).unsqueeze(0)), 0)
                psnry_res = torch.cat((psnry_res, torch.tensor(psnry).unsqueeze(0)), 0)

        # psnrx_res 32,1
        psnr_xy = torch.cat((psnrx_res.unsqueeze(1), psnry_res.unsqueeze(1)),1).float().to("cuda")
        psnr_xy_ac = 1- torch.tanh(psnr_xy)

        l_total = 0
        loss_dict = OrderedDict()

        d_loss = self.de_loss(self.reg, psnr_xy_ac)
        l_total += d_loss
        loss_dict['d_loss'] = d_loss


        # # actprob
        # actprob = torch.gather(self.prob_res, 1, self.flag_res)
        # # actprob 32,1,1,1

        # # reward 
        # for i in range(self.output.size()[0]):
        #     if i == 0:
        #         lossr = F.l1_loss(self.output[i], self.gt[i], reduction='mean').detach().unsqueeze(0)
        #     else:
        #         lox = F.l1_loss(self.output[i], self.gt[i], reduction='mean').detach().unsqueeze(0)
        #         lossr = torch.cat((lossr, lox), 0)
        # # print(lossr)

        # self.flag_ress = actprob.clone()
        # for i in range(self.flag_res.size()[0]):
        #     if self.flag_res[i] == 0:
        #         self.flag_ress[i] = 0.0001
        #     elif self.flag_res[i] == 1:
        #         self.flag_ress[i] = 0.00005
        #     elif self.flag_res[i] == 2:
        #         self.flag_ress[i] = 0

        # reward = -lossr.unsqueeze(1).unsqueeze(2).unsqueeze(3) + self.flag_ress
        # reward = self.calc_reward_to_go(reward)

        # if self.cri_rf:
        #     l_rf = self.cri_rf(actprob, reward)
        #     l_total += l_rf
        #     loss_dict['l_rf'] = l_rf

        # pixel loss
        # if self.cri_pix:
        #     l_pix = self.cri_pix(self.output, self.gt)
        #     l_total += l_pix
        #     loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # input = torch.randn(1, 3, 32, 32).to(self.device)
        # flops, params = profile(self.net_g, inputs=(input))
        # print('FLOPs = ' + str(flops/1000**3) + 'G')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():

                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, self.flag_list = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    print(calculate_metric(metric_data, opt_))
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


    # def calc_reward_to_go(self, reward_list, gamma=0.99):
    #     # range(start, stop, step)

    #     for i in range(len(reward_list) - 2, -1, -1):
    #         # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
    #         reward_list[i] += gamma * reward_list[i + 1]  # Gt

    #     # # 归一化，减去均值除以方差
    #     # reward_list = (reward_list - np.mean(reward_list)) / np.std(reward_list)
    #     return reward_list
