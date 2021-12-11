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

from models.pointrend.point_rend import PointRend 
from datasets.image import ImageDataset
from torch.utils.data import DataLoader
from utils.config import Struct 
from utils.image import overlay_davis
from utils.uncertainty import calculate_uncertainty

torch.set_grad_enabled(False)
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

# Config 
config = Struct(yaml.load(open('/home/nero/MediaEval2021/Medico/Medico/configs/resunet50.yaml', 'r'), Loader=yaml.Loader))
print(config)

dataset_root = '../datasets/Test2021'
dataset = ImageDataset(dataset_root, subset='val')

dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 1, pin_memory=True)

pretrained_path = '/home/nero/MediaEval2021/Medico/Medico/runs/Nov14_18.14.54_medico_ResUneXt50/Nov14_18.14.54_medico_ResUneXt50_10000.pth'

network = PointRend(config).cuda()
print(network.load_state_dict(torch.load(pretrained_path)))

network.eval()
network.eval_()

wandb.init(project='visualize_medico', id = config.DESCRIPTION)

output = 'output/' + config.DESCRIPTION + '/mask'
viz = 'output/' + config.DESCRIPTION + '/viz'
prob = 'output/' + config.DESCRIPTION + '/prob'
palette = Image.open('palette.png').convert('P').getpalette() 

os.makedirs(output, exist_ok=True)
os.makedirs(viz, exist_ok=True)
os.makedirs(prob, exist_ok=True)

# Set path to test dataset
# TEST_DATASET_PATH = "/medico2020"
# MASK_PATH = "/mask"

# torch.cuda.synchronize()
# Model 
total_process_time = 0

# with torch.cuda.amp.autocast(enabled=False):
for i, data in tqdm(enumerate(dataloader), total = len(dataloader)):
    Fs = data['img'].cuda()
    Ms = data['mask'].cuda()
    size = (data['info']['size'][0][0].item(), data['info']['size'][1][0].item())
    name = data['info']['name'][0].replace('.jpg', '.png')
    
    process_begin = time.time()
    logits, history = network(Fs, Ms, 0)
    total_process_time += time.time() - process_begin
    logits = F.softmax(F.interpolate(logits, size = size, mode = 'bilinear', align_corners=False), dim = 1)
    preds = torch.argmax(logits, dim = 1)[0].cpu().numpy().astype(np.uint8)
    # print(preds.shape, preds.dtype)
        
    # Save 
    mask = Image.fromarray(preds * 255).convert('P')
    mask.save(f'{output}/{name}')

    img = (F.interpolate(Fs, size = size, mode = 'bilinear', align_corners=False)[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_davis(img, preds * 2, palette))
    overlay.save(f'{viz}/{name}')

    pr = logits[0, 1].detach().cpu().numpy() * 255
    # print(pr.shape, np.max(pr))
    pr = Image.fromarray(pr.astype(np.uint8)).convert('P')
    pr.save(f'{prob}/{name}')        

    x = f'Test_{i // 20}/{name}'
    wandb.log({x : [
        wandb.Image(img, masks = {
                            "prediction": {
                                "mask_data": preds * 2, 
                                "class_labels": {2: "foreground"}
                            }}),
        wandb.Image(pr, caption = 'probs'),
        wandb.Image(calculate_uncertainty(logits).detach().cpu().numpy()[0], caption = 'probs'),
            ]
        })

total_datasets = len(dataset)
print('Total processing time: ', total_process_time)
print('Total processed images: ', total_datasets)
print('FPS: ', total_datasets / total_process_time)
