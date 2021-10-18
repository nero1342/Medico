import torch 
import yaml 

from utils.config import Struct 
from models.pointrend.point_rend import PointRend
from models.deeplab.deeplab import DeepLab 

config = Struct(yaml.load(open('configs/pointrend.yaml', 'r'), Loader=yaml.Loader))
print(config)

# pointrend = PointRend(config)
# print(pointrend)

input = torch.rand(2, 3, 128, 128, requires_grad=True).cuda()
targets = torch.randint(2, (2, 128, 128)) 

print(input.shape, targets.shape)

deeplab = DeepLab(backbone='resnet', output_stride=16).cuda().eval()

# pointrend.eval()
out1, out2 = deeplab(input)
print(out1.shape, out2.shape)

