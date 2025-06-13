import torch
import torch.nn as nn
from torch.nn import functional as F
from .resnet import *


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        b,c,h,w = inputs1.size()
        inputs2 = F.interpolate(inputs2, size=(h,w), mode='bilinear', align_corners=True)
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes = 21, pretrained = True, backbone = 'resnet18'):
        super(Unet, self).__init__()

        if backbone == "resnet50":
            self.resnet = resnet50(in_channels,pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        elif backbone == "resnet101":
            self.resnet = resnet101(in_channels,pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        elif backbone == "resnet34":
            self.resnet = resnet34(in_channels,pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        elif backbone == "resnet18":
            self.resnet = resnet18(in_channels,pretrained = pretrained)
            in_filters  = [192, 320, 640, 768]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use resnet50.'.format(backbone))
        
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50' or backbone == 'resnet18':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, msi):

        inputs = msi

        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

if __name__ == '__main__':

    model = Unet(in_channels = 2, num_classes = 21, pretrained = True, backbone = 'resnet18')
    x = torch.randn(1, 2, 256, 256)
    y = model(x)
    print(y.shape)