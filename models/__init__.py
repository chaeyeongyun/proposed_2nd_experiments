import torchvision
import torch.nn as nn
from .unet import UNet
from .deeplabv3_plus import DeepLabv3Plus
from .backbones import VGG16, ResNet50 # ResNet101

model_dict = {
    'unet':UNet,
    'deeplabv3_plus': DeepLabv3Plus
}
backbone_dict = {
    'vgg16':VGG16,
    'resnet50':ResNet50,
    # 'resnet101':ResNet101
}
def make_model(backbone:str, seg_head:str, in_channels:int, num_classes:int):
    model = model_dict[seg_head](backbone=backbone, in_channels=in_channels, num_classes=num_classes)
    return model

def __init_weight(feature, init_func:nn, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init_func(m.weight, **kwargs)

        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_weight(module_list, init_func:nn, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, init_func, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, init_func, norm_layer, bn_eps, bn_momentum,
                      **kwargs)