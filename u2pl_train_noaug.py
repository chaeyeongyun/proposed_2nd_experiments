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
import numpy as np

import models
from utils.logging import Logger, save_ckpoints, load_ckpoints
from utils.load_config import get_config_from_json
from utils.env_utils import device_setting
from utils.processing import img_to_label, label_to_onehot, save_result_img
from utils.lr_schedulers import WarmUpPolyLR

from data.dataset import BaseDataset
from data.augmentations import augmentation
from metrics import Measurement
from loss import make_loss, make_unreliable_weight, compute_contra_memobank_loss, cal_unsuploss_weight

# 일단 no cutmix version
def train(cfg):
    print(cfg)
    if cfg.wandb_logging:
        logger_name = cfg.project_name+str(len(os.listdir(cfg.train.save_dir)))
        logger = Logger(cfg, logger_name) if cfg.wandb_logging else None
        wandb.config.update(cfg.train)
        save_dir = os.path.join(cfg.train.save_dir, logger_name)
        os.makedirs(save_dir)
        ckpoints_dir = os.path.join(save_dir, 'ckpoints')
        os.mkdir(ckpoints_dir)
        img_dir = os.path.join(save_dir, 'imgs')
        os.mkdir(img_dir)
        log_txt = open(os.path.join(save_dir, 'log_txt'), 'w')
    
    half=cfg.train.half
    num_classes = cfg.num_classes
    batch_size = cfg.train.batch_size
    num_epochs = cfg.train.num_epochs
    device = device_setting(cfg.train.device)
    measurement = Measurement(num_classes)
    
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
    
    criterion = make_loss(cfg.train.criterion, num_classes, ignore_index=255)
    unsup_loss_weight = cfg.train.unsup_loss.weight
    contra_loss_weight = cfg.train.contrastive_loss.weight
    sup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='labelled', resize=cfg.resize)
    unsup_dataset = BaseDataset(os.path.join(cfg.train.data_dir, 'train'), split='unlabelled', resize=cfg.resize)
    
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    
    
    lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.train.lr_scheduler.lr_power, 
                                total_iters=len(unsup_loader)*num_epochs,
                                warmup_steps=len(unsup_loader)*cfg.train.lr_scheduler.warmup_epoch)
    ema_decay_origin = cfg.train.ema_decay
    # TODO: different lr in encoder and decoder ()
    decoder_lr_times = cfg.train.get("decoder_lr_times", False)
    if decoder_lr_times:
        param_list = []
        backbone = student.backbone
        decoder = student.decoder
        param_list.append(dict(params=backbone.parameters(), lr=cfg.train.learning_rate))
        param_list.append(dict(params=decoder.parameters(), lr=cfg.train.learning_rate*decoder_lr_times))
        optimizer = torch.optim.Adam(param_list, betas=(0.9, 0.999))
        
    else:
        optimizer = torch.optim.Adam(student.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    
    # TODO: build class-wise memory bank 관련 함수 형성
    memobank = []
    queue_ptrlis = []
    queue_size = []

    for i in range(num_classes):
        memobank.append([torch.zeros(0, 64)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            num_classes,
            cfg.train.contrastive_loss.num_queries,
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
            l_input, l_target = sup_dict['img'].to(device), sup_dict['target']
            l_target = img_to_label(l_target, cfg.pixel_to_label).to(device)
            ul_input = unsup_dict['img'].to(device)
            if epoch < cfg.train.sup_only_epoch:
                with torch.cuda.amp.autocast(enabled=half):
                    l_pred_st = student(l_input)
                    sup_loss = criterion(l_pred_st, l_target)
                    unsup_loss = 0 * l_pred_st.sum()
                    contra_loss = 0 * l_pred_st.sum()
                    
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
                # strong augmentation
                if cfg.train.get('strong_aug', False):    
                    ul_input_aug, pseudo_ul_aug, logits_ul_teacher_aug = augmentation(
                        ul_input, pseudo_ul.clone(), logits_ul_teacher.clone(),
                        cfg.train.strong_aug
                    )
                else:
                    ul_input_aug, pseudo_ul_aug, logits_ul_teacher_aug = ul_input, pseudo_ul, logits_ul_teacher
                # supervised loss
                num_label = len(l_input)
                all_input = torch.cat((l_input, ul_input_aug), dim=0)
                with torch.cuda.amp.autocast(enabled=half):
                    # student inference
                    output = student(all_input, return_rep=True)
                    pred_st, rep_st = output['pred'], output['rep']
                    l_pred_st, ul_pred_st = pred_st[:num_label], pred_st[num_label:]
                    sup_loss = criterion(l_pred_st, l_target)
                # teacher forward
                teacher.train()
                with torch.no_grad():
                    out_t = teacher(all_input, return_rep=True)
                    pred_t, rep_t = out_t['pred'], out_t['rep']
                    score_t = F.softmax(pred_t, dim=1)
                    l_score_t, ul_score_t = score_t[:num_label], score_t[:num_label:]
                    ul_pred_t = pred_t[num_label:]
                    
                # TODO: unsupervised_loss
                percent_unreliable = cfg.train.unsup_loss.drop_percent * (1-epoch/num_epochs)
                drop_percent = 100 - percent_unreliable
                unreliable_weight = cal_unsuploss_weight(ul_pred_st, pseudo_ul_aug.clone(), drop_percent, ul_pred_t.detach())
                with torch.cuda.amp.autocast(enabled=half):
                    unsup_loss = unsup_loss_weight*unreliable_weight* criterion(ul_pred_st, pseudo_ul_aug)
                # contrastive loss
                alpha_t = cfg.train.contrastive_loss.low_entropy_threshold * (1-epoch/num_epochs)
                with torch.no_grad():
                    prob = torch.softmax(ul_pred_t, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[pseudo_ul_aug != 255].cpu().numpy().flatten(), alpha_t # entorpy 에서 alpha_t % 에 해당하는 값이 하한값
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * (pseudo_ul_aug != 255).bool() # ndarray.le는 self<=value를 반환한다.
                    )

                    high_thresh = np.percentile(
                        entropy[pseudo_ul_aug != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                        entropy.ge(high_thresh).float() * (pseudo_ul_aug != 255).bool() # ndarray.ge는 self>=value를 반환한다.
                    )

                    low_mask_all = torch.cat(
                        (
                            (l_target.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )
                    # low_mask_all = F.interpolate(
                    #     low_mask_all, size=pred_student.shape[2:], mode="nearest"
                    # )
                    if cfg.train.contrastive_loss.negative_high_entropy:
                        high_mask_all = torch.cat(
                            (
                                (l_target.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        high_mask_all = torch.cat(
                            (
                                (l_target.unsqueeze(1) != 255).float(),
                                torch.ones(logits_ul_teacher_aug.shape)
                                .float()
                                .unsqueeze(1)
                                .cuda(),
                            ),
                        )
                    l_target_onehot = label_to_onehot(l_target, num_classes=num_classes).to(device)
                    pseudo_ul_aug_onehot = label_to_onehot(pseudo_ul_aug,  num_classes=num_classes).to(device)
                    
                with torch.cuda.amp.autocast(enabled=half):
                    prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_st,
                            l_target_onehot.long(),
                            pseudo_ul_aug_onehot.long(),
                            l_score_t.detach(),
                            ul_score_t.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg.train.contrastive_loss,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_t.detach(),
                            prototype,
                            i_iter=epoch * len(unsup_loader) + batch_idx
                        )
                    contra_loss = contra_loss_weight * contra_loss
                    
            with torch.cuda.amp.autocast(enabled=half):
                loss = sup_loss + unsup_loss + contra_loss
            ## learning rate update
            current_idx = epoch * len(unsup_loader) + batch_idx
            learning_rate = lr_scheduler.get_lr(current_idx)
            # update the learning rate
            optimizer.param_groups[0]['lr'] = learning_rate
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # update teacher model with EMA
            if epoch >= cfg.train.sup_only_epoch:
                with torch.no_grad():
                    ema_decay = min(
                        1
                        - 1
                        / (
                            current_idx
                            - len(sup_loader) * cfg.train.sup_only_epoch
                            + 1
                        ),
                        ema_decay_origin,
                    )
                    for t_params, s_params in zip(
                        teacher.parameters(), student.parameters()
                    ):
                        t_params.data = (
                            ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                        )
            
            step_miou, iou_list = measurement.miou(measurement._make_confusion_matrix(l_pred_st.detach().cpu().numpy(), l_target.detach().cpu().numpy()))
            sum_miou += step_miou
            sum_loss += loss.item()
            sum_sup_loss += sup_loss.item()
            sum_unsup_loss += unsup_loss.item()
            sum_cont_loss += contra_loss.item()
            back_iou += iou_list[0]
            weed_iou += iou_list[1]
            crop_iou += iou_list[2]
            print_txt = f"[Epoch{epoch}/{cfg.train.num_epochs}][Iter{batch_idx+1}/{len(unsup_loader)}] lr={learning_rate:.6f}" \
                            + f"miou={step_miou:.4f}, sup_loss={sup_loss:.4f}, unsup_loss_2={unsup_loss:.4f}, contra_loss={contra_loss:.6f}"
            pbar.set_description(print_txt, refresh=False)
            if logger != None: log_txt.write(print_txt)
        
        ## end epoch ## 
        back_iou, weed_iou, crop_iou = back_iou / len(unsup_loader), weed_iou / len(unsup_loader), crop_iou / len(unsup_loader)
        contra_loss = sum_cont_loss / len(unsup_loader)
        sup_loss = sum_sup_loss / len(unsup_loader)
        unsup_loss = sum_unsup_loss / len(unsup_loader)
        loss = sum_loss / len(unsup_loader)
        miou = sum_miou / len(unsup_loader)
        
        if cfg.train.save_img and epoch >= cfg.train.sup_only_epoch and logger!=None:
            # save_result_img(l_input.detach().cpu().numpy(), l_target.detach().cpu().numpy(), ul_input_aug.detach().cpu().numpy(), 
            #                 filename=[f"{epoch}ep_ex{i}.png" for i in range(len(l_input))],
            #                 save_dir=img_dir)
            for i in range(len(ul_input_aug)): 
                cpu_ul_input_aug = ul_input_aug.detach().cpu().numpy()
                cpu_ul_input_aug = cpu_ul_input_aug.transpose(0, 2, 3, 1) #(N H W C)
                plt.imsave(os.path.join(img_dir, f'{epoch}ep_aug_sample_{i}.png'), cpu_ul_input_aug[i])
        print_txt = f"[Epoch{epoch}]" \
                            + f"miou={miou:.4f}, sup_loss={sup_loss:.4f}, unsup_loss_2={unsup_loss:.4f}, contra_loss={contra_loss:.6f}, loss:{loss:.4f}"
        if logger!=None: log_txt.write(print_txt)
        if epoch % 10 == 0 and logger != None:
            torch.save({"student":student.state_dict(),
                        "teacher":teacher.state_dict(),
                        "epoch":epoch,
                        "batch_idx":batch_idx,
                        "optimizer":optimizer.state_dict()},
                        os.path.join(ckpoints_dir, f"{epoch}ep.pth"))
        # wandb logging
        if logger is not None:
            for key in logger.config_dict.keys():
                logger.config_dict[key] = eval(key)
            
            for key in logger.log_dict.keys():
                logger.log_dict[key] = eval(key)
            
            logger.logging(epoch=epoch)
            logger.config_update()
            
    if logger != None: log_txt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/train/u2pl_train_barlow.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    cfg.train.pop('strong_aug')
    train(cfg)