import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from utils import img_to_label

class SemiSupDataset(Dataset):
    def __init__(self, data_dir) :
        super().__init__()
        self.img_dir = os.path.join(data_dir, 'input')
        self.target_dir = os.path.join(data_dir, 'target')
        all_imgs = os.listdir(self.img_dir)
        self.targets = os.listdir(os.path.join(data_dir, 'target'))
        self.unlab_imgs = list(set(all_imgs) - set(self.targets)) # unlabelled images
    def __len__(self):
        return len(self.targets) + len(self.unlab_imgs)
    def __getitem__(self, index) :
        target = self.targets[index]
        unlab_img = Image.open(os.path.join(self.img_dir, self.unlab_imgs[index])).convert('RGB')
        sup_img = Image.open(os.path.join(self.img_dir, target)).convert('RGB')
        sup_target = Image.open(os.path.join(self.target_dir, target)).convert('L')
        # to tensor
        unlab_img = TF.to_tensor(unlab_img)
        sup_img = TF.to_tensor(unlab_img)
        sup_target = torch.from_numpy(np.array(sup_target))
        
        # transforms
        # if self.randomaug:
            # aug = random.randint(0, 7)
            # if aug==1:
            #     input_img = input_img.flip(1)
            #     target_img = target_img.flip(1)
            # elif aug==2:
            #     input_img = input_img.flip(2)
            #     target_img = target_img.flip(2)
            # elif aug==3:
            #     input_img = torch.rot90(input_img,dims=(1,2))
            #     target_img = torch.rot90(target_img,dims=(1,2))
            # elif aug==4:
            #     input_img = torch.rot90(input_img,dims=(1,2), k=2)
            #     target_img = torch.rot90(target_img,dims=(1,2), k=2)
            # elif aug==5:
            #     input_img = torch.rot90(input_img,dims=(1,2), k=-1)
            #     target_img = torch.rot90(target_img,dims=(1,2), k=-1)
            # elif aug==6:
            #     input_img = torch.rot90(input_img.flip(1),dims=(1,2))
            #     target_img = torch.rot90(target_img.flip(1),dims=(1,2))
            # elif aug==7:
            #     input_img = torch.rot90(input_img.flip(2),dims=(1,2))
            #     target_img = torch.rot90(target_img.flip(2),dims=(1,2))
        return {"sup_img":sup_img, "sup_target":sup_target, "unlab_img":unlab_img}