from typing import Dict ,List
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import VGG16, ResNet50

class BasicConv(nn.Sequential):
    """
    Basic Conv Block : conv2d - batchnorm - act
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size, stride:int=1, padding:int=0, dilation=1, groups=1, bias=True, norm='batch', act:nn.Module=nn.ReLU()):
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
        if norm=='instance':
            modules += [nn.InstanceNorm2d(out_channels)]
        if norm=='batch':
            modules += [nn.BatchNorm2d(out_channels)]
        if act is not None:
            modules += [act]
        super().__init__(*modules)
        
### intermediate layer feature getter from resnet50
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers:Dict):
        """Module wrapper that returns intermediate layers from a model

        Args:
            model : model on which we will extract the features
            return_layers (Dict[name, return_name]): 
                a dictionary containing the names of modules for which the activations will be returned as the key of the dict, 
                and value of the dict is the name of the returned activation
        """
        # return layer로 지정된 이름의 레이어가 model 내에 없으면 에러 발생
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        
        self.return_layers = copy.deepcopy(return_layers)       
        # ResNet class로 생성되어있는 모델 객체를 모듈이름:모듈클래스로 이뤄진 OrderedDict()로 변형
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            # return_layer를 모두 포함할 때까지 반복
            if name in return_layers:
                del return_layers[name]
            if not return_layers: break
        
        super(IntermediateLayerGetter, self).__init__(layers)
    def forward(self, x):
        output = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers: # return_layers key에 name이 있으면
                output[self.return_layers[name]] = x # output name으로 지정한 이름으로 현재 레이어의 출력을 저장
        return output
    
#### ASPP module, deeplab head
class ASPPConv(nn.Sequential):
    ''' Atrous Convolution Layer in ASPP module '''
    def __init__(self, in_channels: int, out_channels: int, dilation: int, kernel_size=3, act=nn.ReLU()):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation, bias=False), # inputsize == outputsize
            nn.BatchNorm2d(out_channels),
        ]
        if act is not None:
            assert isinstance(act, nn.Module), "act must be nn.Module instance"
            modules += [act]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    ''' pooling layer in ASPP module in DeepLap v3 '''
    def __init__(self, in_channels: int, out_channels: int, act=nn.ReLU()):
        super().__init__(
            nn.AdaptiveAvgPool2d(1), # GAP
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            act
        )
        
    def forward(self, x: torch.Tensor):
        size = x.shape[-2:] # (H, W)
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates:List[int], out_channels:int=256 ):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 conv
        modules += [nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())]
        for rate in atrous_rates:
            modules += [ASPPConv(in_channels, out_channels, rate)]

        modules += [ASPPPooling(in_channels, out_channels)]
        self.convlist = nn.ModuleList(modules)
        self.project = nn.Sequential(BasicConv(len(self.convlist) * (out_channels), out_channels, 1, norm='batch', act=nn.ReLU()), nn.Dropout(0.1)) 
        
    def forward(self, x):
        conv_results = []
        for conv in self.convlist:
            conv_results += [conv(x)]
        output = torch.cat(conv_results, dim=1)
        output = self.project(output)
        return output

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate):
        super().__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
#### deeplabv3 plus
class DeepLabv3Plus(nn.Module):
    def __init__(self, backbone:str, in_channels:int, num_classes:int, output_stride=8):
        super().__init__()
        
        if output_stride==8:
            replace_stride_with_dilation=[False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation=[False, False, True]
            aspp_dilate = [6, 12, 18]
        
        if backbone == 'vgg16':
            backbone = VGG16(in_channels=in_channels)
            in_channels = 512
            low_level_channels = 256
            return_layers = {'conv5':'out', 'conv3':'low_level'}
        
        elif backbone == 'resnet50':
            backbone = ResNet50()
            in_channels = 2048
            low_level_channels = 256
            return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.decoder = DeepLabHeadV3Plus(in_channels, low_level_channels, num_classes, aspp_dilate)
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        output = self.decoder(features)
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)
        return output