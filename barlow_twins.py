from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision
import torch
import models

from tqdm import tqdm
import glob
import argparse
import os
import random
from utils.env_utils import device_setting

import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt

class SimpleDataset(Dataset):
    def __init__(self, img_folder, transform):
        super().__init__()
        self.imgs = glob.glob(os.path.join(img_folder, '*.png'))
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        y1, y2 = self.transform(img)
        return y1, y2
            

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
class Transform:
    def __init__(self, resize):
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(512, interpolation=Image.BICUBIC)
            transforms.Resize(resize, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            # transforms.RandomResizedCrop(resize, interpolation=Image.BICUBIC),
            transforms.Resize(resize, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

# https://github.dev/facebookresearch/barlowtwins
class Projector(nn.Module):
    def __init__(self, start_ch, channels_list):
        super().__init__()
        self.avgpool = nn.AdaptiveMaxPool2d(output_size=(1,1))
        layers = []
        for i in channels_list:
            layers.append(nn.Linear(start_ch, i, bias=False))
            layers.append(nn.BatchNorm1d(i))
            layers.append(nn.ReLU())
            start_ch = i    
        self.linears = nn.Sequential(*layers)
    def forward(self, x):
        output = self.avgpool(x).squeeze()
        output = self.linears(output)
        return output 
# def make_projector(start_ch, channel_list):
#     layers = []
#     start_ch = start_ch # vgg는 512로 시작
#     layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
#     for i in channel_list:
#         layers.append(nn.Linear(start_ch, i, bias=False))
#         layers.append(nn.BatchNorm1d(i))
#         layers.append(nn.ReLU())
#         start_ch = i
#     return nn.Sequential(*layers)
    
def main(opt):
    half = opt.half
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    save_dir = os.path.join(opt.save_dir, opt.backbone+'_'+str(len(os.listdir(opt.save_dir))))
    os.makedirs(save_dir)
    ckpoints_dir = os.path.join(save_dir, 'ckpoints')
    os.mkdir(ckpoints_dir)
    
    device = device_setting(opt.device)
    # make model 
    backbone = models.backbone_dict[opt.backbone](in_channels=3).to(device)
    projector = Projector(512, [1024, 2048, 1024])
    projector = projector.to(device)
    bn = nn.BatchNorm1d(1024, affine=False).to(device)
    # dataset
    dataset = SimpleDataset(opt.img_folder, Transform(resize=opt.resize))
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(backbone.parameters(), lr=opt.base_lr, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-8)
    
    scaler = torch.cuda.amp.GradScaler(enabled=half)
    min_loss = 10000
    loss_list = []
    backbone.train()
    bn.train()
    projector.train()
    for epoch in range(num_epochs):
        sum_loss = 0
        for y1, y2 in tqdm(trainloader):
            y1 = y1.to(device)
            y2 = y2.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=half):
                z1 = projector(backbone(y1))
                z2 = projector(backbone(y2))

                # empirical c`ross-correlation matrix
                c = bn(z1).T @ bn(z2)

                # sum the cross-correlation matrix between all gpus
                c.div_(batch_size)
                

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                loss = on_diag + 0.0051 * off_diag
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            sum_loss += loss.item()
        
        epoch_loss = sum_loss / len(trainloader)
        loss_list.append(epoch_loss)
        print(f"[Epoch {epoch}]Loss: {epoch_loss}")
        if epoch_loss <= min_loss:
            min_loss = epoch_loss
            torch.save(backbone.state_dict(), os.path.join(ckpoints_dir, f'min_loss_ep{epoch}.pth'))
        if epoch%10==0:
            torch.save(backbone.state_dict(), os.path.join(ckpoints_dir, f'{epoch}ep.pth'))
        torch.save(backbone.state_dict(), os.path.join(ckpoints_dir, f'model_last.pth'))
    # save loss graph
    plt.figure(figsize=(10,5))
    plt.title('loss')
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_graph.png'))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='vgg16', help='backbone name to train')
    parser.add_argument('--device', type=int, default=0, help='device number')
    parser.add_argument('--img_folder', type=str, default='/content/data/semi_sup_data/CWFID/num30/train/input', help='img data folder')
    parser.add_argument('--num_epochs', type=int, default=500, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--half', type=bool, default=False, help='mixed precision')
    parser.add_argument('--resize', type=int, default=512, help='input resize')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/self_supervised/CWFID', help='path for saving ckpoints')
    
    
    opt = parser.parse_args()
    main(opt)