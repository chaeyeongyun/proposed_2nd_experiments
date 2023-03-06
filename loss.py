import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import List
import numpy as np

from utils.processing import label_to_onehot

def make_loss(loss_name:str, num_classes:int, ignore_index:int):
    loss_dict = {'cross_entropy':nn.CrossEntropyLoss,
                 'dice_loss':DiceLoss,
                 'focal_loss':FocalLoss}
    if loss_name == 'cross_entropy':
        return loss_dict[loss_name](ignore_index=ignore_index)
    else:
        return loss_dict[loss_name](num_classes=num_classes)
    

###### dice loss
def dice_coefficient(pred:torch.Tensor, target:torch.Tensor, num_classes:int):
    """calculate dice coefficient

    Args:
        pred (torch.Tensor): (N, num_classes, H, W)
        target (torch.Tensor): (N, H, W)
        num_classes (int): the number of classes
    """
    
    if num_classes == 1:
        target = target.type(pred.type())
        pred = torch.sigmoid(pred)
        # target is onehot label
    else:
        target = target.type(pred.type()) # target과 pred의 type을 같게 만들어준다.
        target = torch.eye(num_classes)[target.long()].to(pred.device) # (N, H, W, num_classes)
        target = target.permute(0, 3, 1, 2) # (N, num_classes, H, W)
        pred = F.softmax(pred, dim=1)
    
    inter = torch.sum(pred*target, dim=(2, 3)) # (N, num_classes)
    sum_sets = torch.sum(pred+target, dim=(2, 3)) # (N, num_classes)
    dice_coefficient = (2*inter / (sum_sets+1e-6)).mean(dim=0) # (num_classes)
    return dice_coefficient
        
        
def dice_loss(pred, target, num_classes, weights:tuple=None, ignore_index=None):
    if not isinstance(pred, torch.Tensor) :
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(pred)}")

    dice = dice_coefficient(pred, target, num_classes)
    if weights is not None:
        dice_loss = 1-dice
        weights = torch.Tensor(weights)
        dice_loss = dice_loss * weights
        dice_loss = dice_loss.mean()
        
    else: 
        dice = dice.mean()
        dice_loss = 1 - dice
        
    return dice_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weights:tuple=None, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.ignore_index = ignore_index
    def forward(self, pred, target):
        return dice_loss(pred, target, self.num_classes, weights=self.weights, ignore_index=self.ignore_index)

  
## focal loss   
def focal_loss(pred:torch.Tensor, target:torch.Tensor, alpha, gamma, num_classes, ignore_index=None, reduction="sum"):
    assert pred.shape[0] == target.shape[0],\
        "pred tensor and target tensor must have same batch size"
    
    if num_classes == 1:
        pred = F.sigmoid(pred)
    
    else:
        pred = F.softmax(pred, dim=1).float()

    onehot = label_to_onehot(target, num_classes) if target.dim()==3 else target
    focal_loss = 0

    focal = torch.pow((1-pred), gamma) # (B, C, H, W)
    ce = -torch.log(pred) # (B, C, H, W)
    focal_loss = alpha * focal * ce * onehot
    focal_loss = torch.sum(focal_loss, dim=1) # (B, H, W)
    
    if reduction == 'none':
        # loss : (B, H, W)
        loss = focal_loss
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(focal_loss)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(focal_loss)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    
    return loss
    

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, ignore_index=None, reduction='sum'):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        if self.num_classes == 1:
            pred = F.sigmoid(pred)
        else:
            pred = F.softmax(pred, dim=1).float()
        
        return focal_loss(pred, target, self.alpha, self.gamma, self.num_classes, self.ignore_index, self.reduction)

def make_unreliable_weight(pred_1, target, pred_2, percent):
    # student 예측, teacher label, drop percent, teacher pred
    # 이거를 해당 모델 예측, 다른 모델로 만든 pseudo label, 다른 모델의 예측으로 바꾸면 될 듯
    batch_size, num_classes, h, w = pred_1.shape
    with torch.no_grad():
        prob = torch.softmax(pred_2, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        weight = batch_size * h * w / torch.sum(target != 255)
    return weight

def cal_unsuploss_weight(pred, target, percent, pred_teacher):
    batch_size, num_class, h, w = pred.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        weight = batch_size * h * w / torch.sum(target != 255)
    return weight



@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    # gather keys before updating queue
    keys = keys.detach().clone().cpu() # (0, 64)

    batch_size = keys.shape[0]

    ptr = int(queue_ptr)

    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr

    return batch_size
### contrastive loss / memory bank
# def compute_contra_memobank_loss(
#     rep_st,
#     label_l,
#     label_u,
#     prob_l,
#     prob_u,
#     low_mask,
#     high_mask,
#     cfg,
#     memobank,
#     queue_prtlis,
#     queue_size,
#     rep_teacher,
#     momentum_prototype=None,
#     i_iter=0,
# ):
#     # current_class_threshold: delta_p (0.3)
#     # current_class_negative_threshold: delta_n (1)
#     current_class_threshold = cfg.current_class_threshold
#     current_class_negative_threshold = cfg.current_class_negative_threshold
#     low_rank, high_rank = cfg.low_rank, cfg.high_rank
#     temp = cfg.temperature
#     num_queries = cfg.num_queries
#     num_negatives = cfg.num_negatives

#     num_feat = rep_st.shape[1]
#     num_labeled = label_l.shape[0]
#     num_segments = label_l.shape[1]
#     ## entropy 기준으로 mask한 라벨 
#     low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
#     high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

#     rep_st = rep_st.permute(0, 2, 3, 1) # N H W C
#     rep_teacher = rep_teacher.permute(0, 2, 3, 1) # N H W C

#     seg_feat_all_list = []
#     seg_feat_low_entropy_list = []  # candidate anchor pixels -> entropy가 낮음 : 믿을만함
#     seg_num_list = []  # the number of low_valid pixels in each class
#     seg_proto_list = []  # the center of each class

#     _, prob_indices_l = torch.sort(prob_l, 1, True)
#     prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

#     _, prob_indices_u = torch.sort(prob_u, 1, True) # dimension1기준으로, descending=True라서 내림차순으로 정렬
#     prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)  # (num_unlabeled, h, w, num_cls)

#     prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

#     valid_classes = []
#     new_keys = []
#     for i in range(num_segments): # 클래스수만큼
#         # TODO: 여기부터
#         low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
#         high_valid_pixel_seg = high_valid_pixel[:, i]

#         prob_seg = prob[:, i, :, :]
#         rep_mask_low_entropy = (
#             prob_seg > current_class_threshold
#         ) * low_valid_pixel_seg.bool()
#         rep_mask_high_entropy = (
#             prob_seg < current_class_negative_threshold
#         ) * high_valid_pixel_seg.bool()

#         seg_feat_all_list.append(rep_st[low_valid_pixel_seg.bool()])
#         seg_feat_low_entropy_list.append(rep_st[rep_mask_low_entropy])

#         # positive sample: center of the class
#         seg_proto_list.append(
#             torch.mean(
#                 rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
#             )
#         )

#         # generate class mask for unlabeled data
#         # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
#         class_mask_u = torch.sum(
#             prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
#         ).bool()

#         # generate class mask for labeled data
#         # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0)
#         # prob_i_classes = prob_indices_l[label_l_mask]
#         class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

#         class_mask = torch.cat(
#             (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
#         )

#         negative_mask = rep_mask_high_entropy * class_mask

#         keys = rep_teacher[negative_mask].detach()
#         new_keys.append(
#             dequeue_and_enqueue(
#                 keys=keys,
#                 queue=memobank[i],
#                 queue_ptr=queue_prtlis[i],
#                 queue_size=queue_size[i],
#             )
#         )

#         if low_valid_pixel_seg.sum() > 0:
#             seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
#             valid_classes.append(i)

#     if (
#         len(seg_num_list) <= 1
#     ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
#         if momentum_prototype is None:
#             return new_keys, torch.tensor(0.0) * rep_st.sum()
#         else:
#             return momentum_prototype, new_keys, torch.tensor(0.0) * rep_st.sum()

#     else:
#         reco_loss = torch.tensor(0.0).cuda()
#         seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
#         valid_seg = len(seg_num_list)  # number of valid classes

#         prototype = torch.zeros(
#             (prob_indices_l.shape[-1], num_queries, 1, num_feat)
#         ).cuda()

#         for i in range(valid_seg):
#             if (
#                 len(seg_feat_low_entropy_list[i]) > 0
#                 and memobank[valid_classes[i]][0].shape[0] > 0
#             ):
#                 # select anchor pixel
#                 seg_low_entropy_idx = torch.randint(
#                     len(seg_feat_low_entropy_list[i]), size=(num_queries,)
#                 )
#                 anchor_feat = (
#                     seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
#                 )
#             else:
#                 # in some rare cases, all queries in the current query class are easy
#                 reco_loss = reco_loss + 0 * rep_st.sum()
#                 continue

#             # apply negative key sampling from memory bank (with no gradients)
#             with torch.no_grad():
#                 negative_feat = memobank[valid_classes[i]][0].clone().cuda()

#                 high_entropy_idx = torch.randint(
#                     len(negative_feat), size=(num_queries * num_negatives,)
#                 )
#                 negative_feat = negative_feat[high_entropy_idx]
#                 negative_feat = negative_feat.reshape(
#                     num_queries, num_negatives, num_feat
#                 )
#                 positive_feat = (
#                     seg_proto[i]
#                     .unsqueeze(0)
#                     .unsqueeze(0)
#                     .repeat(num_queries, 1, 1)
#                     .cuda()
#                 )  # (num_queries, 1, num_feat)

#                 if momentum_prototype is not None:
#                     if not (momentum_prototype == 0).all():
#                         ema_decay = min(1 - 1 / i_iter, 0.999)
#                         positive_feat = (
#                             1 - ema_decay
#                         ) * positive_feat + ema_decay * momentum_prototype[
#                             valid_classes[i]
#                         ]

#                     prototype[valid_classes[i]] = positive_feat.clone()

#                 all_feat = torch.cat(
#                     (positive_feat, negative_feat), dim=1
#                 )  # (num_queries, 1 + num_negative, num_feat)

#             seg_logits = torch.cosine_similarity(
#                 anchor_feat.unsqueeze(1), all_feat, dim=2
#             )

#             reco_loss = reco_loss + F.cross_entropy(
#                 seg_logits / temp, torch.zeros(num_queries).long().cuda()
#             )

#         if momentum_prototype is None:
#             return new_keys, reco_loss / valid_seg
#         else:
#             return prototype, new_keys, reco_loss / valid_seg
def compute_contra_memobank_loss(
    rep_st,
    label_l,
    label_u,
    prob_l,
    prob_u,
    low_mask,
    high_mask,
    cfg,
    memobank,
    queue_prtlis,
    queue_size,
    rep_teacher,
    momentum_prototype=None,
    i_iter=0,
):
    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    current_class_threshold = cfg.current_class_threshold
    current_class_negative_threshold = cfg.current_class_negative_threshold
    low_rank, high_rank = cfg.low_rank, cfg.high_rank
    temp = cfg.temperature
    num_queries = cfg.num_queries
    num_negatives = cfg.num_negatives

    num_feat = rep_st.shape[1] 
    num_labeled = label_l.shape[0] # N
    num_segments = label_l.shape[1]
    ## entropy 기준으로 mask한 라벨 
    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask # (2N C H W)
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask # (2N C H W)

    rep_st = rep_st.permute(0, 2, 3, 1) # N H W num_cls
    rep_teacher = rep_teacher.permute(0, 2, 3, 1) # N H W num_cls

    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels -> entropy가 낮음 : 믿을만함
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class

    # torch.sort에서 indices는 원래 위치 인덱스를 말한다.
    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True) # dimension1기준으로, descending=True라서 내림차순으로 정렬
    prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)  # (num_unlabeled, h, w, num_cls)

    prob = torch.cat((prob_l, prob_u), dim=0)  # (2N, num_cls, h, w)

    valid_classes = []
    new_keys = []
    for i in range(num_segments): # 클래스수만큼
        # TODO: 여기부터
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class (2N, H, W)
        high_valid_pixel_seg = high_valid_pixel[:, i] # (2N, H, W)

        prob_seg = prob[:, i, :, :] # (2N, H, W)
        rep_mask_low_entropy = (
            prob_seg > current_class_threshold
        ) * low_valid_pixel_seg.bool() # (2N, H, W)
        rep_mask_high_entropy = (
            prob_seg < current_class_negative_threshold
        ) * high_valid_pixel_seg.bool() # (2N, H, W)

        seg_feat_all_list.append(rep_st[low_valid_pixel_seg.bool()])
        # rep_st (2N, H, W, feats) , low_valid_pixel_seg.bool() (2N, H, W), 
        # rep_st[low_valid_pixel_seg.bool()] (1인 픽셀 수, feats)
        seg_feat_low_entropy_list.append(rep_st[rep_mask_low_entropy])

        # positive sample: center of the class
        # 같은 class 내의 모든 anchors에게 positive sample은 같다. anchors의 중심
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            ) # (1, 64)
        )

        # generate class mask for unlabeled data
        class_mask_u = torch.sum(
            prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool() # (N, H, W)
        # unlabelled : [low_rank, high_rank) 범위의 rank를 가지면서 gamma_t보다 큰 entropy를 가지면 qualified negative sample. 여기서는 eq로 i에 해당하는 것을 골라냄
        
        # generate class mask for labeled data
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool() # (N, H, W)
        # labelled : [0, low_rank)의 rank값을 가지며 i 클래스에 속하지 않으면 qualified negative sample이다. -> 여기서는 eq를 통해 그 클래스에 속하는 것을 뽑음
        
        class_mask = torch.cat(
            (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
        ) # (2N, H, W)

        negative_mask = rep_mask_high_entropy * class_mask # (N, H, W)

        keys = rep_teacher[negative_mask].detach()
        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )

        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    if (
        len(seg_num_list) <= 1
    ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep_st.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep_st.sum()

    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
        valid_seg = len(seg_num_list)  # number of valid classes

        prototype = torch.zeros(
            (prob_indices_l.shape[-1], num_queries, 1, num_feat)
        ).cuda()

        for i in range(valid_seg):
            if (
                len(seg_feat_low_entropy_list[i]) > 0
                and memobank[valid_classes[i]][0].shape[0] > 0
            ):
                # select anchor pixel
                seg_low_entropy_idx = torch.randint(
                    len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                )
                anchor_feat = (
                    seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
                )
            else:
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep_st.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)
            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda()

                high_entropy_idx = torch.randint(
                    len(negative_feat), size=(num_queries * num_negatives,)
                )
                negative_feat = negative_feat[high_entropy_idx]
                negative_feat = negative_feat.reshape(
                    num_queries, num_negatives, num_feat
                )
                positive_feat = (
                    seg_proto[i]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(num_queries, 1, 1)
                    .cuda()
                )  # (num_queries, 1, num_feat)

                if momentum_prototype is not None:
                    if not (momentum_prototype == 0).all():
                        ema_decay = min(1 - 1 / i_iter, 0.999)
                        positive_feat = (
                            1 - ema_decay
                        ) * positive_feat + ema_decay * momentum_prototype[
                            valid_classes[i]
                        ]

                    prototype[valid_classes[i]] = positive_feat.clone()

                all_feat = torch.cat(
                    (positive_feat, negative_feat), dim=1
                )  # (num_queries, 1 + num_negative, num_feat)

            seg_logits = torch.cosine_similarity(
                anchor_feat.unsqueeze(1), all_feat, dim=2
            )

            reco_loss = reco_loss + F.cross_entropy(
                seg_logits / temp, torch.zeros(num_queries).long().cuda()
            )

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg
        else:
            return prototype, new_keys, reco_loss / valid_seg