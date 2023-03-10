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
from utils.logging import Logger, save_ckpoints, load_ckpoints
from utils.load_config import get_config_from_json
from utils.env_utils import device_setting
from utils.processing import img_to_label
from utils.lr_schedulers import WarmUpPolyLR

from data.dataset import BaseDataset
from data.augmentations import augmentation
from loss import make_loss
from metrics import Measurement

# 일단 no cutmix version
def train(cfg):
    logger_name = cfg.project_name+str(len(os.listdir(cfg.train.save_dir)))
    if cfg.wandb_logging:
        save_dir = os.path.join(cfg.train.save_dir, logger_name)
        os.makedirs(save_dir)
        ckpoints_dir = os.path.join(save_dir, 'ckpoints')
        os.mkdir(ckpoints_dir)
        img_dir = os.path.join(save_dir, 'imgs')
        os.mkdir(img_dir)
        log_txt = open(os.path.join(save_dir, 'log_txt'), 'w')
        wandb.config.update(cfg.train)
    logger = Logger(cfg, logger_name) if cfg.wandb_logging else None
    
    half=cfg.train.half
    num_classes = cfg.num_classes
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs
    device = device_setting(cfg.train.device)
    measurement = Measurement(num_classes)
    
    model_1 = models.make_model(cfg.model.backbone.name, cfg.model.seg_head.name, cfg.model.in_channels, num_classes).to(device)
    model_2 = models.make_model(cfg.model.backbone.name, cfg.model.seg_head.name, cfg.model.in_channels, num_classes).to(device)
    
    # initialize differently (segmentation head)
    if cfg.train.init_weights:
        models.init_weight([model_1.decoder], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
        models.init_weight([model_2.decoder], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
    
    # load encoder pretrain weights
    if cfg.model.backbone.pretrain_weights != None: # if you don't want to use pretrain weights, set cfg.model.backbone.pretrain_weights to null
        model_1.backbone.load_state_dict(torch.load(cfg.model.backbone.pretrain_weights))
        model_2.backbone.load_state_dict(torch.load(cfg.model.backbone.pretrain_weights))
    
    criterion = make_loss(cfg.train.criterion, num_classes, ignore_index=255)
    
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
    cps_loss_weight = cfg.train.cps_loss_weight
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    for epoch in range(num_epochs):
        trainloader = iter(zip(cycle(sup_loader), unsup_loader))
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_cps_loss, sum_sup_loss_1, sum_sup_loss_2 = 0, 0, 0
        sum_loss = 0
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
            ul_input = ul_input.to(device)
            # estimate pseudo label
            with torch.no_grad():
                logits_ul_1 = model_1(ul_input)
                logits_ul_1 = logits_ul_1.detach()
                
                logits_ul_2 = model_2(ul_input)
                logits_ul_2 = logits_ul_2.detach()
                
                pseudo_1, pseudo_2 = torch.argmax(logits_ul_1, dim=1).long(), torch.argmax(logits_ul_2, dim=1).long()
            ul_input_aug, pseudo_aug_1, _ = augmentation(ul_input, pseudo_1, logits_ul_1, cfg.train.strong_aug)
            ul_input_aug, pseudo_aug_2, _ = augmentation(ul_input, pseudo_2, logits_ul_2,  cfg.train.strong_aug)
            
            with torch.cuda.amp.autocast(enabled=half):
                ## predict in unsupervised manner ##
                pred_ul_1 = model_1(ul_input_aug)
                pred_ul_2 = model_2(ul_input_aug)
                cps_loss = criterion(pred_ul_1, pseudo_aug_2) + criterion(pred_ul_2, pseudo_aug_1)
                
            with torch.cuda.amp.autocast(enabled=half):
                pred_sup_1 = model_1(l_input)
                pred_sup_2 = model_2(l_input)
                sup_loss_1 = criterion(pred_sup_1, l_target)
                sup_loss_2 = criterion(pred_sup_2, l_target)
                sup_loss = sup_loss_1 + sup_loss_2    
                
                ## learning rate update
                current_idx = epoch * len(unsup_loader) + batch_idx
                learning_rate = lr_scheduler.get_lr(current_idx)
                # update the learning rate
                optimizer_1.param_groups[0]['lr'] = learning_rate
                optimizer_2.param_groups[0]['lr'] = learning_rate
                
                loss = sup_loss + cps_loss_weight*cps_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer_1)
            scaler.step(optimizer_2)
            scaler.update()
            
            
            step_miou, iou_list = measurement.miou(measurement._make_confusion_matrix(pred_sup_1.detach().cpu().numpy(), l_target.detach().cpu().numpy()))
            sum_miou += step_miou
            sum_loss += loss.item()
            sum_cps_loss += cps_loss.item()
            sum_sup_loss_1 += sup_loss_1.item()
            sum_sup_loss_2 += sup_loss_2.item()
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(unsup_loader)}] lr={learning_rate:.2f}" \
                            + f"miou={step_miou}, sup_loss_1={sup_loss_1:.4f}, sup_loss_2={sup_loss_2:.4f}, cps_loss={cps_loss:.4f}"
            pbar.set_description(print_txt, refresh=False)
            if cfg.wandb_logging:
                log_txt.write(print_txt)
        
        ## end epoch ## 
        back_iou, weed_iou, crop_iou = back_iou / len(unsup_loader), weed_iou / len(unsup_loader), crop_iou / len(unsup_loader)
        cps_loss = sum_cps_loss / len(unsup_loader)
        sup_loss_1 = sum_sup_loss_1 / len(unsup_loader)
        sup_loss_2 = sum_sup_loss_2 / len(unsup_loader)
        loss = loss / len(unsup_loader)
        miou = sum_miou / len(unsup_loader)
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou=miou, sup_loss_1={sup_loss_1:.4f}, sup_loss_2={sup_loss_2:.4f}, cps_loss={cps_loss:.4f}"
        if logger != None:
            log_txt.write(print_txt)
            for key in logger.config_dict.keys():
                logger.config_dict[key] = eval(key)
            
            for key in logger.log_dict.keys():
                logger.log_dict[key] = eval(key)
            
            logger.logging(epoch=epoch)
            logger.config_update()
            
            if cfg.train.save_img:
                for i in range(len(ul_input_aug)): 
                    cpu_ul_input_aug = ul_input_aug.detach().cpu().numpy()
                    cpu_ul_input_aug = cpu_ul_input_aug.transpose(0, 2, 3, 1) #(N H W C)
                    plt.imsave(os.path.join(img_dir, f'{epoch}ep_aug_sample_{i}.png'), cpu_ul_input_aug[i])
            
            if epoch % 10 == 0 :
                save_ckpoints(model_1.state_dict(),
                            model_2.state_dict(),
                            epoch,
                            batch_idx,
                            optimizer_1.state_dict(),
                            optimizer_2.state_dict(),
                            os.path.join(ckpoints_dir, f"{epoch}ep.pth"))
         
    if logger != None:
        log_txt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/train/vgg16_unet_csp_cutmix.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    
    train(cfg)