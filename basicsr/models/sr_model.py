import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

# from basicsr.archs import build_network
# from basicsr.losses import build_loss
# from basicsr.metrics import calculate_metric
# from basicsr.utils import get_root_logger, imwrite, tensor2img
# from basicsr.utils.registry import MODEL_REGISTRY
# from .base_model import BaseModel

import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

# @MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = define_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

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
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # # define losses
        # if train_opt.get('pixel_opt'):
        #     self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        # else:
        #     self.cri_pix = None

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)      #如何写 weighted loss 呢？传参构造Loss函数
        else:
            raise ValueError('pixel loss are None.')

        # if train_opt.get('perceptual_opt'):
        #     self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        if train_opt.get('perceptual_opt'):
            pixel_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
            self.cri_perceptual = cri_perceptual_cls(**train_opt['perceptual_opt']).to(
                self.device)      #如何写 weighted loss 呢？传参构造Loss函数
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
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
    
    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)


        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
        # hello qcc~

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
                # print(self.lq.shape)
                # print("asdfauiwefhauwefuawefiuhauiew")
                # print(self.output.shape)
            self.net_g.train()

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                  mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        # window_size = self.opt['val'].get('window_size', 0)
        test = self.test
        # if window_size:
        #     # test = partial(self.pad_test, window_size)
        #     test = self.test
        # else:
        #     test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric:<6}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    # def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
    #     # Set a fixed width for the metric column and value column for better alignment
    #     log_str = f'Validation {dataset_name}\n'
    #     for metric, value in self.metric_results.items():
    #         log_str += f'\t # {metric:<6}: {value:.4f}'  # Align metric names to the left with a width of 6 characters
    #         if hasattr(self, 'best_metric_results'):
    #             log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
    #                         f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
    #         log_str += '\n'

    #     logger = get_root_logger()
    #     logger.info(log_str)
    #     if tb_logger:
    #         for metric, value in self.metric_results.items():
    #             tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)
    
    
    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer
    
    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)
