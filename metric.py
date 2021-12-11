import numpy as np 
import torch 

def calculate_f_score(precision, recall, base=2):
    return (1 + base**2)*(precision*recall)/((base**2 * precision) + recall)
    
def calculate_all_metrics(preds, gt, cpu=True):
    '''
    Args:
        preds, gt: [batch, C, H, W], here C = 1 for mask
    Return:
        {
            'dice_coeff': np.array[dice-coeff scores]
            'IoU': np.array[IoU scores]
            'precision': np.array[Precision scores]
            'recall': np.array[Recall scores]
            'F2': np.array[F2 scores]
        }
    '''

    batch_size = preds.size(0)
    y_preds = preds.view(batch_size, -1)
    y_true = gt.view(batch_size, -1)
    smooth = 1.0
    eps = 1e-5
    intersection = torch.sum(y_preds*y_true, dim=1)
    union = torch.sum(y_preds, 1) + torch.sum(y_true,1) - intersection

    iou = intersection/(union + eps)
    dice_coeff = (2.0*intersection + smooth)/(y_preds.sum(1) + y_true.sum(1) + smooth)
    precision = intersection / (y_true.sum(1)  + eps)
    recall = intersection / (y_preds.sum(1) + eps)
    # f2 = calculate_f_score(precision, recall, 2)

    all_scores = {
        'dice_coeff': dice_coeff.mean(),
        'IoU': iou.mean(),
        'precision': precision.mean(),
        'recall': recall.mean(),
        # 'F2': f2
    }

    # if cpu == True:
    #     for k in all_scores.keys():
    #         all_scores[k] = all_scores[k].cpu().numpy()

    return all_scores

import os 
from PIL import Image
from utils.integrator import Integrator
from utils.logger import PrintLogger
from pprint import pprint 
from tqdm import tqdm 
logger = PrintLogger()
integrator = Integrator(logger, distributed=False, local_rank=0, world_size=1)

gt = '/home/nero/MediaEval2021/Medico/datasets/Test2020/masks/' 
pred = 'output2020/Ensemble/mask/' 

lst = sorted(os.listdir(gt) )

for img in tqdm(lst):
    # print(img)
    Ms = (torch.from_numpy(np.array(Image.open(gt + img).convert('P'))).unsqueeze(0) / 128).int()
    preds = (torch.from_numpy(np.array(Image.open(pred + img.split('.')[0] + '.png').convert('P'))).unsqueeze(0) / 255).int()
    metrics = calculate_all_metrics(preds, Ms)
    # pprint(metrics)
    integrator.add_dict(metrics)
    # break 
integrator.finalize('test', 0)
integrator.reset_except_hooks()
