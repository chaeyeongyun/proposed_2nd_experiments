import os
import argparse
from glob import glob
from tqdm import tqdm
from typing import List
import numpy as np


import torch
from torch.utils.data import DataLoader
from utils.load_config import get_config_from_json
import torch.nn.functional as F

import models
from utils.processing import img_to_label, save_result_img
from utils.env_utils import device_setting
from data.dataset import BaseDataset
from metrics import Measurement

def test_folder(cfg):
    num_classes = cfg.num_classes
    batch_size = cfg.test.batch_size
    # make directories to save results
    save_dir = os.path.join(cfg.test.save_dir, cfg.model.backbone.name+cfg.model.seg_head.name+'_'+str(len(os.listdir(cfg.test.save_dir))))
    os.makedirs(save_dir)
    img_dir = os.path.join(save_dir, 'imgs')
    os.mkdir(img_dir)
   
    device = device_setting(cfg.test.device)
    model = models.make_model(cfg.model.backbone.name, cfg.model.seg_head.name, cfg.model.in_channels, num_classes).to(device)
   
    test_data = BaseDataset(os.path.join(cfg.test.data_dir, 'test'), split='labelled', resize=cfg.resize)
    testloader = DataLoader(test_data, 1, shuffle=False)
    
    measurement = Measurement(num_classes)
    f = open(os.path.join(save_dir, 'results.txt'), 'w')
    f.write(f"data_dir:{cfg.test.data_dir}, weights:{cfg.test.weights}, save_dir:{cfg.test.save_dir}")

    weights_list = glob(os.path.join(cfg.test.weights, '*.pth'))
    best_miou = 0
    for weights in weights_list:
        model.load_state_dict(torch.load(weights)['model_1'])
        model.eval()
        test_acc, test_miou = 0, 0
        test_precision, test_recall, test_f1score = 0, 0, 0
        iou_per_class = np.array([0]*num_classes, dtype=np.float64)
        for data in tqdm(testloader):
            input_img, mask_img, filename = data['img'], data['target'], data['filename']
            input_img = input_img.to(device)
            mask_cpu = img_to_label(mask_img, cfg.pixel_to_label)
            
            with torch.no_grad():
                pred = model(input_img)
            
            pred = F.interpolate(pred, mask_img.shape[-2:], mode='bilinear')
            pred_cpu, mask_cpu = pred.detach().cpu().numpy(), mask_cpu.cpu().numpy()
            
            acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(pred_cpu, mask_cpu) 
            
            test_acc += acc_pixel
            test_miou += batch_miou
            iou_per_class += iou_ndarray
            
            test_precision += precision
            test_recall += recall
            test_f1score += f1score
                
            
            input_img = F.interpolate(input_img.detach().cpu(), mask_img.shape[-2:], mode='bilinear')
            save_result_img(input_img.numpy(), mask_img.detach().cpu().numpy(), pred.detach().cpu().numpy(), filename, img_dir,
                            colormap = np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]]))
        
        # test finish
        test_acc = test_acc / len(testloader)
        test_miou = test_miou / len(testloader)
        test_ious = np.round((iou_per_class / len(testloader)), 5).tolist()
        test_precision /= len(testloader)
        test_recall /= len(testloader)
        test_f1score /= len(testloader)
        if best_miou <= test_miou:
            best_miou = test_miou
            result_txt = "load model(.pt) : %s \n Testaccuracy: %.8f, Test miou: %.8f" % (weights,  test_acc, test_miou)       
            result_txt += f"\niou per class {test_ious}"
            result_txt += f"\nprecision : {test_precision}, recall : {test_recall}, f1score : {test_f1score} "
    
    print(result_txt)
    f.write(result_txt)
    f.close()
        
def test(cfg):
    num_classes = cfg.num_classes
    batch_size = cfg.test.batch_size
    # make directories to save results
    save_dir = os.path.join(cfg.test.save_dir, cfg.model.backbone.name+cfg.model.seg_head.name+'_'+str(len(os.listdir(cfg.test.save_dir))))
    os.makedirs(save_dir)
    img_dir = os.path.join(save_dir, 'imgs')
    os.mkdir(img_dir)
    
    device = device_setting(cfg.test.device)
    model = models.make_model(cfg.model.backbone.name, cfg.model.seg_head.name, cfg.model.in_channels, num_classes).to(device)
    model.load_state_dict(torch.load(cfg.test.weights)['model_1'])
    test_data = BaseDataset(os.path.join(cfg.test.data_dir, 'test'), split='labelled', resize=cfg.resize)
    testloader = DataLoader(test_data, 1, shuffle=False)
    
    measurement = Measurement(num_classes)
    f = open(os.path.join(save_dir, 'results.txt'), 'w')
    f.write(f"data_dir:{cfg.test.data_dir}, weights:{cfg.test.weights}, save_dir:{cfg.test.save_dir}")

    
    model.eval()
    test_acc, test_miou = 0, 0
    test_precision, test_recall, test_f1score = 0, 0, 0
    iou_per_class = np.array([0]*num_classes, dtype=np.float64)
    for data in tqdm(testloader):
        input_img, mask_img, filename = data['img'], data['target'], data['filename']
        input_img = input_img.to(device)
        mask_cpu = img_to_label(mask_img, cfg.pixel_to_label)
        
        with torch.no_grad():
            pred = model(input_img)
        
        pred = F.interpolate(pred, mask_img.shape[-2:], mode='bilinear')
        pred_cpu, mask_cpu = pred.detach().cpu().numpy(), mask_cpu.cpu().numpy()
        
        acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(pred_cpu, mask_cpu) 
        
        test_acc += acc_pixel
        test_miou += batch_miou
        iou_per_class += iou_ndarray
        
        test_precision += precision
        test_recall += recall
        test_f1score += f1score
            
        
        input_img = F.interpolate(input_img.detach().cpu(), mask_img.shape[-2:], mode='bilinear')
        save_result_img(input_img.numpy(), mask_img.detach().cpu().numpy(), pred.detach().cpu().numpy(), filename, img_dir,
                        colormap = np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]]))
    
    # test finish
    test_acc = test_acc / len(testloader)
    test_miou = test_miou / len(testloader)
    test_ious = np.round((iou_per_class / len(testloader)), 5).tolist()
    test_precision /= len(testloader)
    test_recall /= len(testloader)
    test_f1score /= len(testloader)
    
    result_txt = "load model(.pt) : %s \n Testaccuracy: %.8f, Test miou: %.8f" % (cfg.test.weights,  test_acc, test_miou)       
    result_txt += f"\niou per class {test_ious}"
    result_txt += f"\nprecision : {test_precision}, recall : {test_recall}, f1score : {test_f1score} "
    print(result_txt)

    f.write(result_txt)
    f.close()

def main(cfg):
    if os.path.isfile(cfg.test.weights):
        test(cfg)
    elif os.path.isdir(cfg.test.weights):
        test_folder(cfg)
    else:
        raise ValueError("It's not available path")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config/test/vgg16_unet_cwfid_test.json')
    opt = parser.parse_args()
    cfg = get_config_from_json(opt.config_path)
    
    main(cfg)
   