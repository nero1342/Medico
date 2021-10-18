import sys
sys.path.append('.')
sys.path.append('..')

import yaml
from models.engine.train_manager import Trainer
from utils.config import Struct 

import torch.multiprocessing as mp
import torch 

import torch.distributed as dist

def main_worker(gpu, cfg):
    # Initiate a training manager
    trainer = Trainer(local_rank=gpu, cfg=cfg)
    # Start Training
    trainer.training()
    print("Done.", gpu)

def main():    
    import argparse
    parser = argparse.ArgumentParser(description="Test MSTCN")
    parser.add_argument('--config', type=str, default='configs/pointrend.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    config = Struct(yaml.load(open(args.config, 'r'), Loader=yaml.Loader))
    main_worker(args.local_rank, config)

if __name__ == '__main__':
    main()