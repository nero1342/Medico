import torch 
import yaml 

from utils.config import Struct 
from models.pointrend.point_rend import PointRend
from models.deeplab.deeplab import DeepLab 

config = Struct(yaml.load(open('configs/pointrend.yaml', 'r'), Loader=yaml.Loader))
print(config)

pointrend = PointRend(config)
print(pointrend)

input = torch.rand(2, 3, 512, 512, requires_grad=True)
targets = torch.randint(2, (2, 512, 512)) 

print(input.shape, targets.shape)

pointrend.eval()
out1, out2 = pointrend(input, targets)
print(out1.shape)

