
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# from networks.mstcn.norm import FrozenBatchNorm2d
from models.deeplab.modules import *
class Encoder(nn.Module):
    def __init__(self, config = 'resnet101', BatchNorm = nn.BatchNorm2d):
        super().__init__()
        resnet = getattr(models, config)(
            replace_stride_with_dilation=[False, False, True],
            pretrained=True, norm_layer = BatchNorm)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024
        self.layer4 = resnet.layer4
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, f):
        f = (f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024
        f16 = self.layer4(f16) # 1/16, 1024
        return f16, f8, f4

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.compress = ResBlock(2048, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, num_classes, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)
        p = self.pred(F.relu(x))

        return p

class Network(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=num_classes) 

    def forward(self, input):
        f16, f8, f4 = self.encoder(input)
        x = self.decoder(f16, f8, f4)
        return x, f4
