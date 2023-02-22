import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import ResNet38, ResNet38Light, VGG16
    
class DBConv(nn.Sequential):
    '''
    double 3x3 conv layers with Batch normalization and ReLU
    '''
    def __init__(self, in_channels, out_channels):
        conv_layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(DBConv, self).__init__(*conv_layers)
        
class UNetDecoder(nn.Module):
    def __init__(self, enc_out_channels, num_classes):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        c1, c2, c3, c4, c5 = enc_out_channels[:]
        self.conv1 = DBConv(c5, c5//2)
        self.conv2 = DBConv(c4+c5//2, c4//2)
        self.conv3 = DBConv(c3+c4//2, c3//2)
        self.conv4 = DBConv(c2+c3//2, c2//2)
        self.conv5 = DBConv(c1+c2//2, c2//2)
        
        self.classifier = nn.Conv2d(c2//2, num_classes, kernel_size=1, bias=False)
        
    def forward(self, x, pass1, pass2, pass3, pass4):
        x = self.conv1(x)
        x = torch.cat((self.upsample(x), pass4), dim=1)
        x = self.conv2(x)
        x = torch.cat((self.upsample(x), pass3), dim=1)
        x = self.conv3(x)
        x = torch.cat((self.upsample(x), pass2), dim=1)
        x = self.conv4(x)
        x = torch.cat((self.upsample(x), pass1), dim=1)
        x = self.conv5(x)
        output = self.classifier(x)
        return output
        
        
class UNet(nn.Module):
    def __init__(self, backbone, in_channels, num_classes):
        super().__init__()
        if backbone == 'resnet38':
            self.backbone = ResNet38(in_channels)
            enc_out_channels = [64, 128, 256, 512, 4096]
            self.out_keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv6']
        elif backbone == 'resnet38_light':
            self.backbone = ResNet38Light(in_channels)
            enc_out_channels = self.backbone.enc_out_channels
            self.out_keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv6']
        elif backbone == 'vgg16':
            self.backbone = VGG16(in_channels)
            enc_out_channels = self.backbone.enc_out_channels
            self.out_keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        self.decoder = UNetDecoder(enc_out_channels, num_classes)
    
    def forward(self, x):
        enc_out = [self.backbone.forward_as_dict(x)[key] for key in self.out_keys]
        output = self.decoder(enc_out[-1], *enc_out[:4])
        return output