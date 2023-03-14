import torch
from torch import nn
import numpy as np
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import torchvision

# ResNet 50
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

### ResNet backbone
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet50(nn.Module):
    def __init__(self,  num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrained=False):
        super(ResNet50, self).__init__()
        ## resnet50
        layers=[3, 4, 6, 3]
        block=Bottleneck
        ###
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                              progress=True)
            self.load_state_dict(state_dict)
            
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ResNet 38 https://github.com/peihan-miao/ResNet38-Semantic-Segmentation/blob/master/network/resnet38d.py
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
#         super(ResBlock, self).__init__()

#         self.same_shape = (in_channels == out_channels and stride == 1)

#         if first_dilation == None: first_dilation = dilation

#         self.bn_branch2a = nn.BatchNorm2d(in_channels)

#         self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
#                                        padding=first_dilation, dilation=first_dilation, bias=False)

#         self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

#         self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

#         if not self.same_shape:
#             self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

#     def forward(self, x, get_x_bn_relu=False):

#         branch2 = self.bn_branch2a(x)
#         branch2 = F.relu(branch2)

#         x_bn_relu = branch2

#         if not self.same_shape:
#             branch1 = self.conv_branch1(branch2)
#         else:
#             branch1 = x

#         branch2 = self.conv_branch2a(branch2)
#         branch2 = self.bn_branch2b1(branch2)
#         branch2 = F.relu(branch2)
#         branch2 = self.conv_branch2b1(branch2)

#         x = branch1 + branch2

#         if get_x_bn_relu:
#             return x, x_bn_relu

#         return x

#     def __call__(self, x, get_x_bn_relu=False):
#         return self.forward(x, get_x_bn_relu=get_x_bn_relu)

# class ResBlock_bot(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
#         super(ResBlock_bot, self).__init__()

#         self.same_shape = (in_channels == out_channels and stride == 1)

#         self.bn_branch2a = nn.BatchNorm2d(in_channels)
#         self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

#         self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
#         self.dropout_2b1 = torch.nn.Dropout2d(dropout)
#         self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

#         self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
#         self.dropout_2b2 = torch.nn.Dropout2d(dropout)
#         self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

#         if not self.same_shape:
#             self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
#         else :
#             self.conv_branch1 = nn.Identity()
#     def forward(self, x, get_x_bn_relu=False):

#         branch2 = self.bn_branch2a(x)
#         branch2 = F.relu(branch2)
#         x_bn_relu = branch2

#         branch1 = self.conv_branch1(branch2)

#         branch2 = self.conv_branch2a(branch2)

#         branch2 = self.bn_branch2b1(branch2)
#         branch2 = F.relu(branch2)
#         branch2 = self.dropout_2b1(branch2)
#         branch2 = self.conv_branch2b1(branch2)

#         branch2 = self.bn_branch2b2(branch2)
#         branch2 = F.relu(branch2)
#         branch2 = self.dropout_2b2(branch2)
#         branch2 = self.conv_branch2b2(branch2)

#         x = branch1 + branch2

#         if get_x_bn_relu:
#             return x, x_bn_relu

#         return x

#     def __call__(self, x, get_x_bn_relu=False):
#         return self.forward(x, get_x_bn_relu=get_x_bn_relu)


# class ResNet38(nn.Module):
#     def __init__(self, in_channels):
#         super(ResNet38, self).__init__()
#         self.conv1a = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)

#         self.b2 = ResBlock(64, 128, 128, stride=2)
#         self.b2_1 = ResBlock(128, 128, 128)
#         self.b2_2 = ResBlock(128, 128, 128)

#         self.b3 = ResBlock(128, 256, 256, stride=2)
#         self.b3_1 = ResBlock(256, 256, 256)
#         self.b3_2 = ResBlock(256, 256, 256)

#         self.b4 = ResBlock(256, 512, 512, stride=2)
#         self.b4_1 = ResBlock(512, 512, 512)
#         self.b4_2 = ResBlock(512, 512, 512)
#         self.b4_3 = ResBlock(512, 512, 512)
#         self.b4_4 = ResBlock(512, 512, 512)
#         self.b4_5 = ResBlock(512, 512, 512)

#         self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
#         self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
#         self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

#         self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

#         self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

#         self.bn7 = nn.BatchNorm2d(4096)

#         # self.not_training = [self.conv1a]

        
#         self.not_training = []

#     def forward(self, x):
#         return self.forward_as_dict(x)['conv6']

#     def forward_as_dict(self, x):

#         x = self.conv1a(x)
#         conv1 = x
#         x = self.b2(x) # down
#         x = self.b2_1(x)
#         x = self.b2_2(x)
#         conv2 = x
        
#         x = self.b3(x) # down
#         x = self.b3_1(x)
#         x = self.b3_2(x)
#         conv3 = x
        
#         x = self.b4(x) # down
#         x = self.b4_1(x)
#         x = self.b4_2(x)
#         x = self.b4_3(x)
#         x = self.b4_4(x)
#         x = self.b4_5(x)

#         x, conv4 = self.b5(x, get_x_bn_relu=True)
#         x = self.b5_1(x)
#         x = self.b5_2(x)

#         x, conv5 = self.b6(x, get_x_bn_relu=True)

#         x = self.b7(x)
#         conv6 = F.relu(self.bn7(x))

#         return dict({'conv1':conv1, 'conv2':conv2, 'conv3':conv3, 'conv4': conv4, 'conv5': conv5, 'conv6': conv6})


#     def train(self, mode=True, frozen_batchnorm=False):

#         super().train(mode)

#         for layer in self.not_training:

#             if isinstance(layer, torch.nn.Conv2d):
#                 layer.weight.requires_grad = False

#             elif isinstance(layer, torch.nn.Module) :
#                 for c in layer.children():
#                     c.weight.requires_grad = False
#                     if c.bias is not None:
#                         c.bias.requires_grad = False

#         for layer in self.modules():
#             if isinstance(layer, torch.nn.BatchNorm2d) and frozen_batchnorm:
#                 layer.eval()
#                 layer.bias.requires_grad = False
#                 layer.weight.requires_grad = False
                

# class ResNet38Light(nn.Module):
#     def __init__(self, in_channels):
#         super(ResNet38Light, self).__init__()
#         first_ch = 32
#         self.conv1a = nn.Conv2d(in_channels, first_ch, 3, padding=1, bias=False)

#         self.b2 = ResBlock(first_ch, first_ch*2, first_ch*2, stride=2)
#         self.b2_1 = ResBlock(first_ch*2, first_ch*2, first_ch*2)
#         self.b2_2 = ResBlock(first_ch*2, first_ch*2, first_ch*2)

#         self.b3 = ResBlock(first_ch*2, first_ch*4, first_ch*4, stride=2)
#         self.b3_1 = ResBlock(first_ch*4, first_ch*4, first_ch*4)
#         self.b3_2 = ResBlock(first_ch*4, first_ch*4, first_ch*4)

#         self.b4 = ResBlock(first_ch*4, first_ch*8, first_ch*8, stride=2)
#         self.b4_1 = ResBlock(first_ch*8, first_ch*8, first_ch*8)
#         self.b4_2 = ResBlock(first_ch*8, first_ch*8, first_ch*8)
#         self.b4_3 = ResBlock(first_ch*8, first_ch*8, first_ch*8)
#         self.b4_4 = ResBlock(first_ch*8, first_ch*8, first_ch*8)
#         self.b4_5 = ResBlock(first_ch*8, first_ch*8, first_ch*8)

#         self.b5 = ResBlock(first_ch*8, first_ch*8, first_ch*8, stride=1, first_dilation=1, dilation=2)
#         self.b5_1 = ResBlock(first_ch*8, first_ch*8, first_ch*8, dilation=2)
#         self.b5_2 = ResBlock(first_ch*8, first_ch*8, first_ch*8, dilation=2)

#         self.b6 = ResBlock_bot(first_ch*8, first_ch*8, stride=1, dilation=4, dropout=0.3)

#         self.b7 = ResBlock_bot(first_ch*8, first_ch*8, dilation=4, dropout=0.5)

#         self.bn7 = nn.BatchNorm2d(first_ch*8)

#         # self.not_training = [self.conv1a]
#         self.enc_out_channels = [first_ch, first_ch*2, first_ch*4, first_ch*8, first_ch*8]
        
#         self.not_training = []

#     def forward(self, x):
#         return self.forward_as_dict(x)['conv6']

#     def forward_as_dict(self, x):

#         x = self.conv1a(x)
#         conv1 = x
#         x = self.b2(x) # down
#         x = self.b2_1(x)
#         x = self.b2_2(x)
#         conv2 = x
        
#         x = self.b3(x) # down
#         x = self.b3_1(x)
#         x = self.b3_2(x)
#         conv3 = x
        
#         x = self.b4(x) # down
#         x = self.b4_1(x)
#         x = self.b4_2(x)
#         x = self.b4_3(x)
#         x = self.b4_4(x)
#         x = self.b4_5(x)

#         x, conv4 = self.b5(x, get_x_bn_relu=True)
#         x = self.b5_1(x)
#         x = self.b5_2(x)

#         x, conv5 = self.b6(x, get_x_bn_relu=True)

#         x = self.b7(x)
#         conv6 = F.relu(self.bn7(x))

#         return dict({'conv1':conv1, 'conv2':conv2, 'conv3':conv3, 'conv4': conv4, 'conv5': conv5, 'conv6': conv6})


## VGG16
class VGG16(nn.Module):
    def __init__(self, in_channels=3):
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

if __name__ == '__main__':
    vgg16 = VGG16(3)
    res50 = ResNet50()
    a = 0