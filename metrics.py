import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Measurement:
    def __init__(self, num_classes:int, ignore_idx=None) :
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
    
    def _make_confusion_matrix(self, pred:np.ndarray, target:np.ndarray):
        """make confusion matrix

        Args:
            pred (numpy.ndarray): segmentation model's prediction score matrix
            target (numpy.ndarray): label
            num_classes (int): the number of classes
        """
        assert pred.shape[0] == target.shape[0], "pred and target ndarray's batchsize must have same value"
        N = pred.shape[0]
        # prediction score to label
        pred_label = pred.argmax(axis=1) # (N, H, W)
        
        pred_1d = np.reshape(pred_label, (N, -1)) # (N, HxW)
        target_1d = np.reshape(target, (N, -1)) # (N, HxW)
        # num_classes * gt + pred = category
        cats = self.num_classes * target_1d + pred_1d # (N, HxW)
        conf_mat = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.num_classes**2), axis=-1, arr=cats) # (N, 9)
        conf_mat = np.reshape(conf_mat, (N, self.num_classes, self.num_classes)) # (N, 3, 3)
        return conf_mat
    
    def accuracy(self, pred, target):
        '''
        Args:
            pred: (N, C, H, W), ndarray
            target : (N, H, W), ndarray
        Returns:
            the accuracy per pixel : acc(int)
        '''
        N = pred.shape[0]
        pred = pred.argmax(axis=1) # (N, H, W)
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2])) # (N, HxW)
        target = np.reshape(target, (target.shape[0], target.shape[1]*target.shape[2])) # (N, HxW)
        
        if self.ignore_idx != None:
             not_ignore_idxs = np.where(target != self.ignore_idx) # where target is not equal to ignore_idx
             pred = pred[not_ignore_idxs]
             target = target[not_ignore_idxs]
             
        return np.mean(np.sum(pred==target, axis=-1)/pred.shape[-1])
    
    def miou(self, conf_mat:np.ndarray):
        iou_list = []
        sum_col = np.sum(conf_mat, -2) # (N, 3)
        sum_row = np.sum(conf_mat, -1) # (N, 3)
        for i in range(self.num_classes):
            batch_mean_iou = np.mean(conf_mat[:, i, i] / (sum_col[:, i]+sum_row[:, i]-conf_mat[:, i, i]+1e-8))
            iou_list.append(batch_mean_iou)
        iou_ndarray = np.array(iou_list)
        miou = np.mean(iou_ndarray)
        return miou, iou_list
    
    def precision(self, conf_mat:np.ndarray):
        # confmat shape (N, self.num_classes, self.num_classes)
        sum_col = np.sum(conf_mat, -2)# (N, 3) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        precision_per_class = np.mean(np.array([conf_mat[:, i, i]/ sum_col[:, i] for i in range(self.num_classes)]), axis=-1) # list안에 (N, )가 클래스개수만큼.-> (3, N) -> 평균->(3,)
        # multi class에 대해 recall / precision을 구할 때에는 클래스 모두 합쳐 평균을 낸다.
        mprecision = np.mean(precision_per_class)
        return mprecision, precision_per_class

    def recall(self, conf_mat:np.ndarray):
        # confmat shape (N, self.num_classes, self.num_classes)
        sum_row = np.sum(conf_mat, -1)# (N, 3) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        recall_per_class = np.mean(np.array([conf_mat[:, i, i]/ sum_row[:, i] for i in range(self.num_classes)]), axis=-1) # list안에 (N, )가 클래스개수만큼.-> (3, N) -> 평균->(3,)
        mrecall = np.mean(recall_per_class)
        return mrecall, recall_per_class
    
    def f1score(self, recall, precision):
        return 2*recall*precision/(recall + precision)
    
    def measure(self, pred:np.ndarray, target:np.ndarray):
        conf_mat = self._make_confusion_matrix(pred, target)
        acc = self.accuracy(pred, target)
        miou, iou_list = self.miou(conf_mat)
        precision, _ = self.precision(conf_mat)
        recall, _ = self.recall(conf_mat)
        f1score = self.f1score(recall, precision)
        return acc, miou, iou_list, precision, recall, f1score
        
    __call__ = measure

######## like CEDNet ############
class MeasurementCEDNet:
    def __init__(self, total_num_classes:int=2):
        assert total_num_classes in [2,3], "It's implemented for only crop/weed case or background/crop/weed case"
        self.total_num_classes = total_num_classes
    
    def _make_confusion_matrix(self, pred_1d:np.ndarray, target_1d:np.ndarray, num_classes:int, N):
        """make confusion matrix

        Args:
            pred (numpy.ndarray): segmentation model's prediction score matrix
            target (numpy.ndarray): label
            num_classes (int): the number of classes
        """
        assert pred_1d.shape[0] == target_1d.shape[0], "pred and target ndarray's batchsize must have same value"
        # num_classes * gt + pred = category
        cats = self.total_num_classes * target_1d + pred_1d # (N, HxW)
        conf_mat = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_classes**2), axis=-1, arr=cats) # (N, 9)
        conf_mat = np.reshape(conf_mat, (N, num_classes, num_classes)) # (N, 3, 3)
        conf_mat = np.sum(conf_mat, axis=0)
        return conf_mat
    
    def conf_mat_for_miou(self, pred:np.ndarray, target:np.ndarray):
        # 0: BG, 1: weed, 2:crop
        N = pred.shape[0]
        # prediction score to label
        pred_label = pred.argmax(axis=1) # (N, H, W)
        pred_1d = np.reshape(pred_label, (N, -1)) # (N, HxW)
        target_1d = np.reshape(target, (N, -1)) # (N, HxW)
        
        # crop back binary mask ( crop=1, BG,weed =0 )
        crop_back = np.where(pred_1d==1, 0, pred_1d) # 0, 2만 존재
        crop_back = np.where(crop_back==2, 1, crop_back) # 0, 1 binary로
        crop_back_target = np.where(target_1d==1, 0, target_1d)
        crop_back_target = np.where(crop_back_target==2, 1, crop_back_target)
        
        # weed back binary mask ( weed=1, BG,crop=0 )
        weed_back = np.where(pred_1d==2, 0, pred_1d) # 0, 1만 존재
        weed_back_target = np.where(target_1d==2, 0, target_1d)
        crop_conf = self._make_confusion_matrix(crop_back, crop_back_target, num_classes=2, N=N)
        weed_conf = self._make_confusion_matrix(weed_back, weed_back_target, num_classes=2, N=N)
        total_conf = crop_conf + weed_conf
        
        if self.total_num_classes == 3:
            back_obj = np.where(pred_1d==1, 2, pred_1d) # 0, 2만 존재
            back_obj = np.where(back_obj==0, 1, back_obj) # 1, 2만 존재
            back_obj = np.where(back_obj==2, 0, back_obj) # 0, 1만 존재
            back_obj_target = np.where(target_1d==1, 2, target_1d) # 0, 2만 존재
            back_obj_target = np.where(back_obj_target==0, 1, back_obj_target) # 1, 2만 존재
            back_obj_target = np.where(back_obj_target==2, 0, back_obj_target) # 0, 1만 존재
            back_conf = self._make_confusion_matrix(back_obj, back_obj_target, num_classes=2, N=N)
            total_conf += back_conf
            return total_conf, back_conf, weed_conf, crop_conf
        return total_conf, weed_conf, crop_conf
        