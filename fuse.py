import torch
import torch.nn.functional as F
import yaml
from pprint import pprint
from tqdm import tqdm 
import time 
import numpy as np 
import random 
import os 
from PIL import Image 
import wandb  

from datasets.image import ImageDataset
from torch.utils.data import DataLoader
from utils.config import Struct 
from utils.image import overlay_davis
from utils.uncertainty import calculate_uncertainty

torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

path = 'output'
lst = [
    'ResUneXt101_CBAM_POINTREND',
    'ResUneSt101_CBAM_POINTREND',
    'EfnNetB6_CBAM_POINTREND',
    'ResUneXt50_CBAM_POINTREND',
    'ResUneXt50'
]


DESCRIPTION = 'Ensemble'
wandb.init(project='visualize_medico', id = DESCRIPTION)

output = 'output/' + DESCRIPTION + '/mask'
viz = 'output/' + DESCRIPTION + '/viz'
prob = 'output/' + DESCRIPTION + '/prob'

palette = Image.open('palette.png').convert('P').getpalette() 

os.makedirs(output, exist_ok=True)
os.makedirs(viz, exist_ok=True)
os.makedirs(prob, exist_ok=True)



dataset_root = '../datasets/Test2021'
dataset = ImageDataset(dataset_root, subset='val')

dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 2, pin_memory=True)

for i, data in tqdm(enumerate(dataloader), total = len(dataloader)):
    Fs = data['img'].cuda()
    Ms = data['mask'].cuda()
    size = (data['info']['size'][0][0].item(), data['info']['size'][1][0].item())
    name = data['info']['name'][0].replace('.jpg', '.png')
    print(data['info'], size)
    probs = torch.from_numpy(np.array([np.array(Image.open(os.path.join(path, desc, 'prob', name))) for desc in lst])).unsqueeze(1).float()
    probs = F.interpolate(probs, size = size, mode = 'bilinear', align_corners=False).mean(0)[0].numpy() / 255
    # print(probs.shape, probs.min(), probs.max())

    preds = (probs >= 0.5).astype(np.uint8)     
    # break 
    # Save 
    mask = Image.fromarray(preds * 255).convert('P')
    mask.save(f'{output}/{name}')

    # img = (F.interpolate(Fs, size = size, mode = 'bilinear', align_corners=False)[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    # overlay = Image.fromarray(overlay_davis(img, preds * 2, palette))
    # overlay.save(f'{viz}/{name}')

    # pr = probs * 255
    # # print(pr.shape, np.max(pr))
    # pr = Image.fromarray(pr.astype(np.uint8)).convert('P')
    # pr.save(f'{prob}/{name}')        

    # x = f'Test_{i // 20}/{name}'
    # wandb.log({x : [
    #     wandb.Image(img, masks = {
    #                         "prediction": {
    #                             "mask_data": preds * 2, 
    #                             "class_labels": {2: "foreground"}
    #                         }}),
    #     wandb.Image(pr, caption = 'probs'),
    #     # wandb.Image(calculate_uncertainty(logits).detach().cpu().numpy()[0], caption = 'probs'),
    #         ]
    #     })

