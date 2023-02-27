import argparse
import matplotlib.pyplot as plt
import os
from itertools import cycle
from tqdm import tqdm
import time
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from utils.logging import Logger, save_ckpoints
from utils.load_config import get_config_from_json
from utils.env_utils import device_setting
from utils.processing import img_to_label
from utils.lr_schedulers import WarmUpPolyLR

from data.dataset import BaseDataset
from data.augmentations import CutMix
from loss import make_loss
from metrics import Measurement

# 일단 no cutmix version
def train(cfg):
    save_dir = os.path.join(cfg.train.save_dir, cfg.project_name+str(len(os.listdir(cfg.train.save_dir))))
    os.makedirs(save_dir)
    ckpoints_dir = os.path.join(save_dir, 'ckpoints')
    os.mkdir(ckpoints_dir)
    log_txt = open(os.path.join(save_dir, 'log_txt'), 'w')
    
    half = cfg.train.half
    logger = Logger(cfg) if cfg.wandb_logging else None
    num_classes = cfg.num_classes
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs
    device = device_setting(cfg.train.device)
    measurement = Measurement(num_classes)
    
    model_1 = models.make_model(cfg.model.backbone.name, cfg.model.seg_head.name, cfg.model.in_channels, num_classes).to(device)
    model_2 = models.make_model(cfg.model.backbone.name, cfg.model.seg_head.name, cfg.model.in_channels, num_classes).to(device)
    # TODO: load pretrained weights (only backbone)
    
    # initialize differently (segmentation head)
    if cfg.train.init_weights:
        models.init_weight([model_1.decoder], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
        models.init_weight([model_2.decoder], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
    if cfg.model.backbone.pretrain_weights != None: # if you don't want to use pretrain weights, set cfg.model.backbone.pretrain_weights to null
        model_1.backbone.load_state_dict(torch.load(cfg.model.backbone.pretrain_weights))
        model_2.backbone.load_state_dict(torch.load(cfg.model.backbone.pretrain_weights))
    
    criterion = make_loss(cfg.train.criterion, num_classes)
    
    sup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled', resize=cfg.resize)
    unsup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='unlabelled', resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    
    lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.train.lr_scheduler.lr_power, 
                                total_iters=len(unsup_loader)*num_epochs,
                                warmup_steps=len(unsup_loader)*cfg.train.lr_scheduler.warmup_epoch)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    
    # progress bar
    
    
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    for epoch in range(num_epochs):
        trainloader = iter(zip(cycle(sup_loader), unsup_loader))
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_cps_loss, sum_sup_loss_1, sum_sup_loss_2 = 0, 0, 0
        sum_miou = 0
        ep_start = time.time()
        pbar =  tqdm(range(len(unsup_loader)))
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
            with torch.cuda.amp.autocast(enabled=half):
                pred_sup_1 = model_1(l_input)
                pred_sup_2 = model_2(l_input)
            ## predict in unsupervised manner ##
            
            with torch.cuda.amp.autocast(enabled=half):
                ## supervised loss
                sup_loss_1 = criterion(pred_sup_1, l_target)
                sup_loss_2 = criterion(pred_sup_2, l_target)
                sup_loss = sup_loss_1 + sup_loss_2
            scaler.scale(sup_loss).backward()
            scaler.step(optimizer_1)
            scaler.step(optimizer_2)
            scaler.update()

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            
            ul_input = ul_input.to(device)
            with torch.cuda.amp.autocast(enabled=half):
                pred_ul_1 = model_1(ul_input)
                pred_ul_2 = model_2(ul_input)
                
            ## cps loss ##
            # pseudo label
            pseudo_1 = torch.argmax(pred_ul_1, dim=1).long()
            pseudo_2 = torch.argmax(pred_ul_2, dim=1).long()
            with torch.cuda.amp.autocast(enabled=half):
                ## cps loss
                cps_loss = criterion(pred_ul_1, pseudo_2) + criterion(pred_ul_2, pseudo_1)
            
            scaler.scale(cps_loss).backward()
            scaler.step(optimizer_1)
            scaler.step(optimizer_2)
            scaler.update()
            
            
            ## learning rate update
            current_idx = epoch * len(unsup_loader) + batch_idx
            learning_rate = lr_scheduler.get_lr(current_idx)
            # update the learning rate
            optimizer_1.param_groups[0]['lr'] = learning_rate
            optimizer_2.param_groups[0]['lr'] = learning_rate
            
            step_miou, iou_list = measurement.miou(measurement._make_confusion_matrix(pred_sup_1.detach().cpu().numpy(), l_target.detach().cpu().numpy()))
            sum_miou += step_miou
            sum_cps_loss += cps_loss.item()
            sum_sup_loss_1 += sup_loss_1.item()
            sum_sup_loss_2 += sup_loss_2.item()
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(unsup_loader)}] lr={learning_rate:.2f}" \
                            + f"miou={step_miou}, sup_loss_1={sup_loss_1:.4f}, sup_loss_2={sup_loss_2:.4f}, cps_loss={cps_loss:.4f}"
            pbar.set_description(print_txt, refresh=False)
            log_txt.write(print_txt)
        
        ## end epoch ## 
        back_iou, weed_iou, crop_iou = back_iou / len(unsup_loader), weed_iou / len(unsup_loader), crop_iou / len(unsup_loader)
        
        loss = cps_loss + sup_loss_1 + sup_loss_2
        miou = sum_miou / len(unsup_loader)
        back_iou += iou_list[0]
        weed_iou += iou_list[1]
        crop_iou += iou_list[2]
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou=miou, sup_loss_1={sup_loss_1:.4f}, sup_loss_2={sup_loss_2:.4f}, cps_loss={cps_loss:.4f}"
        log_txt.write(print_txt)
        if epoch % 10 == 0:
            save_ckpoints(model_1.state_dict(),
                          model_2.state_dict(),
                          epoch,
                          batch_idx,
                          optimizer_1.state_dict(),
                          optimizer_2.state_dict(),
                          os.path.join(ckpoints_dir, f"{epoch}ep.pth"))
        # wandb logging
        if logger is not None:
            for key in logger.config_dict.keys():
                logger.config_dict[key] = eval(key)
            
            for key in logger.log_dict.keys():
                logger.log_dict[key] = eval(key)
            
            logger.logging(epoch=epoch)
            logger.config_update()
    log_txt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/train/vgg16_unet_csp_trainver2_barlow.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    
    train(cfg)