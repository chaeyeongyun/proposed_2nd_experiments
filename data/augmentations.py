from typing import Iterable
import torch
import numpy as np
import random
#TODO: 
def augmentation(input:torch.Tensor, label:torch.Tensor, logits:torch.Tensor, aug_cfg:dict):
    batch_size = input.shape[0]
    input_aug, label_aug, logits_aug = [], [], []
    for i in range(batch_size):
        if aug_cfg.name=='cutout':
            mask = make_cutout_mask(input.shape[-2:], aug_cfg.ratio).to(input.device)
            label[i][(1-mask).bool()] = 255 # ignore index
            input_aug.append((input[i]*mask).unsqueeze(0))
            label_aug.append(label[i].unsqueeze(0))
            logits_aug.append((logits[i]*mask).unsqueeze())
        elif aug_cfg.name == 'cutmix':
            mask = make_cutout_mask(input.shape[-2:], aug_cfg.ratio).to(input.device)
            input_aug.append((input[i]*mask + input[(i+1)%batch_size]*(1-mask)).unsqueeze(0))
            label_aug.append((label[i]*mask + label[(i+1)%batch_size]*(1-mask)).unsqueeze(0))
            logits_aug.append((logits[i]*mask + logits[(i+1)%batch_size]*(1-mask)).unsqueeze(0))
    input_aug = torch.cat(input_aug, dim=0)
    label_aug = torch.cat(label_aug, dim=0)
    logits_aug = torch.cat(logits_aug, dim=0)

    return input_aug, label_aug, logits_aug

def make_cutout_mask(img_size:Iterable[int], ratio):
    cutout_area = img_size[0]*img_size[1]*ratio
    cut_w = np.random.randint(int(img_size[1]*ratio)+1, img_size[1])
    cut_h = int(cutout_area//cut_w)
    x1, y1 = np.random.randint(0, img_size[1]-cut_w+1), random.randint(0, img_size[0]-cut_h+1)
    mask = torch.ones(tuple(img_size))
    mask[y1:y1+cut_h, x1:x1+cut_w] = 0
    return mask.long()
    
    
# class CutMix():
#     def __init__(self, bbox_size:Iterable[int]):
#         """CutMix augmentation

#         Args:
#             bbox_size (Iterable[int]): size of bounding box to be cut off. (h, w)
#         """
#         self.bbox_size = bbox_size
        
#     def _make_mask(self, img_size:Iterable[int], bbox_size:Iterable[int]):
#         x1, y1 = random.randint(0, img_size[1]-bbox_size[1]), random.randint(0, img_size[0]-bbox_size[0])
#         mask = torch.zeros(img_size)
#         mask[y1:y1+bbox_size[0], x1:x1+bbox_size[1]] = 1
#         return mask
    
#     def __call__(self, img1:torch.Tensor, img2:torch.Tensor):
#         """
#         Args:
#             img1 (torch.Tensor): the image where bounding box will be cut
#             img2 (torch.Tensor): the image to which bounding box will be attached
#         """
#         assert img1.shape == img2.shape, "It's not implemented for this case yet"
#         h, w = img1.shape[-2:]
#         mask = self._make_mask((h,w), self.bbox_size) # (H, W)
#         mask = torch.stack([mask]*img1.shape[1], dim=0) # (C, H, W)
#         mask = torch.stack([mask]*img1.shape[0], dim=0) # (N, C, H, W)
#         mixed_img = img1 * mask + img2 * (1 - mask)
#         return mixed_img

    