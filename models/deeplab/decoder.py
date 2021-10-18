import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.modules import * 

class Decoder(nn.Module):
    def __init__(self, backbone, BatchNorm, num_classes = 2):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU(inplace=True)
        
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(inplace=True),
                                       nn.Sequential(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(inplace=True),
                                       nn.Sequential())


        self.compress = ResBlock(2048, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.predictor = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self._init_weight()

    def forward(self, f16, f8, f4):
        # low_level_feat = self.conv1(low_level_feat)
        # low_level_feat = self.bn1(low_level_feat)
        # low_level_feat = self.relu(low_level_feat)

        # print(x.shape)
        # x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        # x = torch.cat((x, low_level_feat), dim=1)
        # x = self.last_conv(x)
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)
        x = self.predictor(F.relu(x))
        return x

    # def forward(self, x, low_level_feat):
    #     low_level_feat = self.conv1(low_level_feat)
    #     low_level_feat = self.bn1(low_level_feat)
    #     low_level_feat = self.relu(low_level_feat)

    #     print(x.shape)
    #     x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
    #     x = torch.cat((x, low_level_feat), dim=1)
    #     x = self.last_conv(x)

    #     x = self.predictor(x)
    #     return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(backbone, BatchNorm, num_classes):
    return Decoder(backbone, BatchNorm, num_classes)
