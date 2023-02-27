import torch
import numpy as np
import random
#TODO: 
def augmentation(input, label, logits, mode_list):
    
    return input_aug, logits_aug, label_aug

class CutMix():
    def __init__(self, bbox_size:tuple):
        """CutMix augmentation

        Args:
            bbox_size (tuple): size of bounding box to be cut off. (h, w)
        """
        self.bbox_size = bbox_size
        
    def _make_mask(self, img_size:tuple, bbox_size:tuple):
        x1, y1 = random.randint(0, img_size[1]-bbox_size[1]), random.randint(0, img_size[0]-bbox_size[0])
        mask = torch.zeros(img_size)
        mask[y1:y1+bbox_size[0], x1:x1+bbox_size[1]] = 1
        return mask
    
    def __call__(self, img1:torch.Tensor, img2:torch.Tensor):
        """
        Args:
            img1 (torch.Tensor): the image where bounding box will be cut
            img2 (torch.Tensor): the image to which bounding box will be attached
        """
        assert img1.shape == img2.shape, "It's not implemented for this case yet"
        h, w = img1.shape[-2:]
        mask = self._make_mask((h,w), self.bbox_size) # (H, W)
        mask = torch.stack([mask]*img1.shape[1], dim=0) # (C, H, W)
        mask = torch.stack([mask]*img1.shape[0], dim=0) # (N, C, H, W)
        mixed_img = img1 * mask + img2 * (1 - mask)
        return mixed_img

    