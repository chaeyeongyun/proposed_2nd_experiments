import argparse
import matplotlib.pyplot as plt
import os
from itertools import cycle
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from utils.logging import Logger
from utils.load_config import load_config
from utils.env_utils import device_setting

from data.dataset import BaseDataset
from data.augmentations import CutMix
from utils.processing import img_to_label
from utils.lr_schedulers import WarmUpPolyLR
from loss import make_loss

# 일단 no cutmix version
def train(cfg):
    logger = Logger(cfg) if cfg.wandb_logging else None
    num_classes = cfg.num_classes
    device = device_setting(cfg.train.device)
    
    model_1 = models.make_model(cfg.backbone.name, cfg.seg_head.name).to(device)
    model_2 = models.make_model(cfg.backbone.name, cfg.seg_head.name).to(device)
    # TODO: load pretrained weights (only backbone)
    # initialize differently (segmentation head)
    models.init_weight([model_1.decoder], nn.init.kaiming_normal_,
                       nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                       mode='fan_in', nonlinearity='relu')
    models.init_weight([model_2.decoder], nn.init.kaiming_normal_,
                       nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                       mode='fan_in', nonlinearity='relu')
    
    criterion = make_loss(cfg.train.criterion)
    
    sup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled')
    unsup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled')
    
    sup_loader = DataLoader(sup_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    
    trainloader = iter(zip(cycle(sup_loader), unsup_loader))
    lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.train.lr_scheduler.lr_power, 
                                total_iters=len(unsup_loader)*cfg.train.num_epochs,
                                warmup_steps=len(unsup_loader)*cfg.train.lr_scheduler.warmup_epoch)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    
    # progress bar
    pbar =  tqdm(range(len(unsup_loader)))
    cps_weight = cfg.train.cps_weight
    for epoch in range(cfg.train.num_epochs):
        sum_cps_loss, sum_sup_loss_1, sum_sup_loss_2 = 0, 0, 0
        
        ep_start = time.time()
        for batch_idx in pbar:
            sup_dict, unsup_dict = next(trainloader)
            l_input, l_target = sup_dict['img'], sup_dict['target']
            l_target = img_to_label(l_target, cfg.pixel_to_label)
            ul_input = unsup_dict['img']
            ## predict in supervised manner ##
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            l_input = l_input.to(device)
            l_target = l_target.to(device)
            pred_sup_1 = model_1(l_input)
            pred_sup_2 = model_2(l_input)
            ## predict in unsupervised manner ##
            ul_input = ul_input.to(device)
            pred_ul_1 = model_1(ul_input)
            pred_ul_2 = model_2(ul_input)
            
            ## cps loss ##
            pred_1 = torch.cat([pred_sup_1, pred_ul_1], dim=0)
            pred_2 = torch.cat([pred_sup_2, pred_ul_2], dim=0)
            # pseudo label
            pseudo_1 = torch.argmax(pred_1, dim=1).long()
            pseudo_2 = torch.argmax(pred_2, dim=1).long()
            ## cps loss
            cps_loss = criterion(pred_1, pseudo_2) + criterion(pred_2, pseudo_1)
            ## supervised loss
            sup_loss_1 = criterion(pred_sup_1, l_target)
            sup_loss_2 = criterion(pred_sup_2, l_target)
            sup_loss = sup_loss_1 + sup_loss_2
            
            ## learning rate update
            current_idx = epoch * len(unsup_loader) + batch_idx
            lr = lr_scheduler.get_lr(current_idx)
            # update the learning rate
            optimizer_1.param_groups[0]['lr'] = lr
            optimizer_2.param_groups[0]['lr'] = lr
            
            loss = sup_loss + cps_weight*cps_loss
            loss.backward()
            optimizer_1.step()
            optimizer_2.step()
            
            sum_cps_loss += cps_loss
            sum_sup_loss_1 += sup_loss_1
            sum_sup_loss_2 += sup_loss_2
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(unsup_loader)}] lr={lr:.2f}" \
                            + f"sup_loss_1={sup_loss_1:.4f}, sup_loss_2={sup_loss_2:.4f}, cps_loss={cps_loss:.4f}"
            pbar.set_description(print_txt, refresh=False)
            #TODO: logger 업데이트 부분 추가!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_Argument('--config_path', default='./config/resnet38_unet_csp.json')
    opt = parser.parse_args()
    cfg = load_config(opt.config_path)
    
    train(cfg)