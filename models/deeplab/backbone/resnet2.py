import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, config = 'resnet50', BatchNorm = None):
        super().__init__()
        resnet = getattr(models,config)(
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
        low_level_feat = f4

        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024
        f16 = self.layer4(f16)
        return f16, low_level_feat
