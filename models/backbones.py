import torch
from torch import nn
import numpy as np

import torch.nn.functional as F

# ResNet 38 https://github.com/peihan-miao/ResNet38-Semantic-Segmentation/blob/master/network/resnet38d.py
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else :
            self.conv_branch1 = nn.Identity()
    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class ResNet38(nn.Module):
    def __init__(self, in_channels):
        super(ResNet38, self).__init__()
        self.conv1a = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(4096)

        # self.not_training = [self.conv1a]

        
        self.not_training = []

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = self.conv1a(x)
        conv1 = x
        x = self.b2(x) # down
        x = self.b2_1(x)
        x = self.b2_2(x)
        conv2 = x
        
        x = self.b3(x) # down
        x = self.b3_1(x)
        x = self.b3_2(x)
        conv3 = x
        
        x = self.b4(x) # down
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv1':conv1, 'conv2':conv2, 'conv3':conv3, 'conv4': conv4, 'conv5': conv5, 'conv6': conv6})


    def train(self, mode=True, frozen_batchnorm=False):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module) :
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():
            if isinstance(layer, torch.nn.BatchNorm2d) and frozen_batchnorm:
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False
                

class ResNet38Light(nn.Module):
    def __init__(self, in_channels):
        super(ResNet38Light, self).__init__()
        first_ch = 32
        self.conv1a = nn.Conv2d(in_channels, first_ch, 3, padding=1, bias=False)

        self.b2 = ResBlock(first_ch, first_ch*2, first_ch*2, stride=2)
        self.b2_1 = ResBlock(first_ch*2, first_ch*2, first_ch*2)
        self.b2_2 = ResBlock(first_ch*2, first_ch*2, first_ch*2)

        self.b3 = ResBlock(first_ch*2, first_ch*4, first_ch*4, stride=2)
        self.b3_1 = ResBlock(first_ch*4, first_ch*4, first_ch*4)
        self.b3_2 = ResBlock(first_ch*4, first_ch*4, first_ch*4)

        self.b4 = ResBlock(first_ch*4, first_ch*8, first_ch*8, stride=2)
        self.b4_1 = ResBlock(first_ch*8, first_ch*8, first_ch*8)
        self.b4_2 = ResBlock(first_ch*8, first_ch*8, first_ch*8)
        self.b4_3 = ResBlock(first_ch*8, first_ch*8, first_ch*8)
        self.b4_4 = ResBlock(first_ch*8, first_ch*8, first_ch*8)
        self.b4_5 = ResBlock(first_ch*8, first_ch*8, first_ch*8)

        self.b5 = ResBlock(first_ch*8, first_ch*8, first_ch*8, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(first_ch*8, first_ch*8, first_ch*8, dilation=2)
        self.b5_2 = ResBlock(first_ch*8, first_ch*8, first_ch*8, dilation=2)

        self.b6 = ResBlock_bot(first_ch*8, first_ch*8, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(first_ch*8, first_ch*8, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(first_ch*8)

        # self.not_training = [self.conv1a]
        self.enc_out_channels = [first_ch, first_ch*2, first_ch*4, first_ch*8, first_ch*8]
        
        self.not_training = []

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = self.conv1a(x)
        conv1 = x
        x = self.b2(x) # down
        x = self.b2_1(x)
        x = self.b2_2(x)
        conv2 = x
        
        x = self.b3(x) # down
        x = self.b3_1(x)
        x = self.b3_2(x)
        conv3 = x
        
        x = self.b4(x) # down
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv1':conv1, 'conv2':conv2, 'conv3':conv3, 'conv4': conv4, 'conv5': conv5, 'conv6': conv6})


## VGG16
class VGG16(nn.Module):
    def __init__(self, in_channels):
        super(VGG16, self).__init__()
        # input image size (N, 3, 224, 224)
        # after maxpooling layer, h and w are devided by 2 : 224->112->56->28->14->7
        self.in_channels = in_channels
        # there are out_channels and M(maxpool) in self.vgg_cfg 
        self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        first_ch = 64
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, first_ch, kernel_size=3, padding=1), 
                              nn.Conv2d(first_ch, first_ch, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(first_ch, first_ch*2, kernel_size=3, padding=1), 
                              nn.Conv2d(first_ch*2, first_ch*2, kernel_size=3, padding=1))
        self.conv3 = nn.Sequential(nn.Conv2d(first_ch*2, first_ch*4, kernel_size=3, padding=1), 
                              nn.Conv2d(first_ch*4, first_ch*4, kernel_size=3, padding=1),
                              nn.Conv2d(first_ch*4, first_ch*4, kernel_size=3, padding=1))
        self.conv4 = nn.Sequential(nn.Conv2d(first_ch*4, first_ch*8, kernel_size=3, padding=1), 
                              nn.Conv2d(first_ch*8, first_ch*8, kernel_size=3, padding=1),
                              nn.Conv2d(first_ch*8, first_ch*8, kernel_size=3, padding=1))
        self.conv5 = nn.Sequential(nn.Conv2d(first_ch*8, first_ch*8, kernel_size=3, padding=1), 
                              nn.Conv2d(first_ch*8, first_ch*8, kernel_size=3, padding=1),
                              nn.Conv2d(first_ch*8, first_ch*8, kernel_size=3, padding=1))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc_out_channels = [first_ch, first_ch*2, first_ch*4, first_ch*8, first_ch*8]
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.maxpool(output)
        output = self.conv2(output)
        output = self.maxpool(output)
        output = self.conv3(output)
        output = self.maxpool(output)
        output = self.conv4(output)
        output = self.maxpool(output)
        output = self.conv5(output)
        output = self.maxpool(output)

        return output
    
    def forward_as_dict(self, x):
        out_dict = dict()
        output = self.conv1(x)
        out_dict['conv1'] = output
        output = self.maxpool(output)
        
        output = self.conv2(output)
        out_dict['conv2'] = output
        output = self.maxpool(output)
        
        output = self.conv3(output)
        out_dict['conv3'] = output
        output = self.maxpool(output)
        
        output = self.conv4(output)
        out_dict['conv4'] = output
        output = self.maxpool(output)
        
        output = self.conv5(output)
        out_dict['conv5'] = output
        
        
        return out_dict