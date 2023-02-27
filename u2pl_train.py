import argparse
import matplotlib.pyplot as plt
import os
from itertools import cycle
from tqdm import tqdm
import time
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    save_dir = os.path.join(cfg.train.save_dir, cfg.project_name+str(len(os.listdir(cfg.train.save_dir))))
    os.makedirs(save_dir)
    ckpoints_dir = os.path.join(save_dir, 'ckpoints')
    os.mkdir(ckpoints_dir)
    
    half=cfg.train.half
    logger = Logger(cfg) if cfg.wandb_logging else None
    num_classes = cfg.num_classes
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs
    device = device_setting(cfg.train.device)
    measurement = Measurement(num_classes)
    log_txt = open(os.path.join(save_dir, 'log_txt'), 'w')
    
    student = models.make_model(cfg.model.backbone.name, cfg.model.seg_head.name, cfg.model.in_channels, num_classes).to(device)
    teacher = models.make_model(cfg.model.backbone.name, cfg.model.seg_head.name, cfg.model.in_channels, num_classes).to(device)
    # teacher parameters are updated by EMA. not gradient descent
    for p in teacher.parameters():
        p.requires_grad = False

    # initialize differently (segmentation head)
    if cfg.train.init_weights:
        models.init_weight([student.decoder], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
        models.init_weight([teacher.decoder], nn.init.kaiming_normal_,
                        nn.BatchNorm2d, cfg.train.bn_eps, cfg.train.bn_momentum, 
                        mode='fan_in', nonlinearity='relu')
    
    if cfg.model.backbone.pretrain_weights != None: # if you don't want to use pretrain weights, set cfg.model.backbone.pretrain_weights to null
        student.backbone.load_state_dict(torch.load(cfg.model.backbone.pretrain_weights))
        teacher.backbone.load_state_dict(torch.load(cfg.model.backbone.pretrain_weights))
    
    criterion = make_loss(cfg.train.criterion, num_classes)
    
    sup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled', resize=cfg.resize)
    unsup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='unlabelled', resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    
    
    lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.train.lr_scheduler.lr_power, 
                                total_iters=len(unsup_loader)*num_epochs,
                                warmup_steps=len(unsup_loader)*cfg.train.lr_scheduler.warmup_epoch)
    
    # TODO: different lr in encoder and decoder ()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    
    # TODO: build class-wise memory bank 관련 함수 형성
    memobank = []
    queue_ptrlis = []
    queue_size = []

    for i in range(cfg["net"]["num_classes"]):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            cfg["net"]["num_classes"],
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            256,
        )
    ).to(device)
    # scaler for fp16
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    for epoch in range(num_epochs):
        trainloader = iter(zip(cycle(sup_loader), unsup_loader))
        crop_iou, weed_iou, back_iou = 0, 0, 0
        sum_sup_loss, sum_unsup_loss, sum_cont_loss = 0, 0, 0
        sum_loss = 0
        sum_miou = 0
        ep_start = time.time()
        # progress bar
        pbar =  tqdm(range(len(unsup_loader)))
        for batch_idx in pbar:
            sup_dict, unsup_dict = next(trainloader)
            l_input, l_target = sup_dict['img'], sup_dict['target']
            l_target = img_to_label(l_target, cfg.pixel_to_label)
            l_input, l_target = l_input.to(device), l_target.to(device)
            ul_input = unsup_dict['img'].to(device)
            if epoch < cfg.train.sup_only_epoch:
                with torch.cuda.amp.autocast(enabled=half):
                    pred_l = student(l_input)
                    sup_loss = criterion(pred_l, l_target)
            else:
                if epoch == cfg.train.sup_only_epoch:
                    # copy student parameters to teacher
                    with torch.no_grad(): # nograd 상태로 teacher의 student 의 파라미터를 teacher의 파라미터로 복사
                        for t_params, s_params in zip(
                            teacher.parameters(), student.parameters()
                        ):
                            t_params.data = s_params.data
                # generate pseudo label from teacher model
                teacher.eval()
                pred_ul_teacher = teacher(ul_input)
                pred_ul_teacher = F.softmax(pred_ul_teacher, dim=1)
                logits_ul_teacher, pseudo_ul = torch.max(pred_ul_teacher, dim=1)
                # TODO: strong augmentation
                ul_input_aug, logits_ul_teacher_aug, pseudo_ul_aug = augmentation(
                    ul_input, pseudo_ul.clone(), logits_ul_teacher.clone(),
                    cfg.train.strong_aug
                )
                ## cps loss ##
                pred_1 = torch.cat([pred_sup_1, pred_ul_1], dim=0)
                pred_2 = torch.cat([pred_sup_2, pred_ul_2], dim=0)
                # pseudo label
                pseudo_1 = torch.argmax(pred_1, dim=1).long()
                pseudo_2 = torch.argmax(pred_2, dim=1).long()
                
                with torch.cuda.amp.autocast(enabled=half):
                    ## cps loss
                    cps_loss = criterion(pred_1, pseudo_2) + criterion(pred_2, pseudo_1)
                    ## supervised loss
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
        log_txt.write(print_txt)
        if epoch % 10 == 0:
            save_ckpoints(student.state_dict(),
                        teacher.state_dict(),
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
    parser.add_argument('--config_path', default='./config/train/vgg16_unet_csp_barlow.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    
    train(cfg)