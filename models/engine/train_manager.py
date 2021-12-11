import os
import time
import datetime as datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np 
import math 
from os import path 
import torch.nn.functional as F 
from tqdm import tqdm 
from pathlib import Path 

# Model
from models.pointrend.point_rend import PointRend 
from utils.integrator import Integrator
from utils.learning import adjust_learning_rate
from utils.logger import WandB

# Dataset
from torch.utils.data import DataLoader, ConcatDataset
from datasets.image import ImageDataset
from utils.uncertainty import calculate_uncertainty

# from torchmetrics.functional import accuracy, f1, iou

import random
dist.init_process_group(backend = 'nccl')
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()

def worker_init_fn(worker_id): 
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank * 100)

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


class Trainer(object):
    def __init__(self, cfg, local_rank = 0):
        # Config 
        self.gpu = local_rank
        self.local_rank = local_rank
        self.cfg = cfg
        self.print_log(cfg)
        print("Use GPU {} for training".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        
        torch.manual_seed(14159265)
        np.random.seed(14159265)
        random.seed(14159265)
        
        # Model
        print("Building model....")
        self.network = PointRend(cfg = cfg).cuda()
        if cfg.DIST.ENABLE:
            # Set seed
            self.dist_model = torch.nn.parallel.DistributedDataParallel(
                self.network, 
                device_ids=[self.gpu], broadcast_buffers=False,
                output_device=local_rank,
                find_unused_parameters=True
            )
        else:
            self.dist_model = self.network
        self.network = self.dist_model

        # Logger
        print("Building logger...")
        self.logger = None 
        if self.local_rank == 0:
            long_id = '%s_%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), cfg.ID, cfg.DESCRIPTION)
            self.logger =  WandB(project_name = cfg.ID, save_path=cfg.LOG.SAVE_PATH, config = cfg, id=long_id)
            self.save_path = path.join(cfg.LOG.SAVE_PATH, long_id, long_id)

        # Losses & Metrics
        self.integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=cfg.TRAIN.GPUS)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.network.parameters()), 
            lr=cfg.OPTIMIZER.LR, 
            # momentum=0.9,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
        )

        # Scheduler 
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, cfg.SCHEDULER.STEPS, cfg.SCHEDULER.GAMMA)
        self.base_lr = self.cfg.OPTIMIZER.LR 
        #   
        self.prepare_dataset()
        self.process_pretrained_model()

        if cfg.AMP:
            self.scaler = torch.cuda.amp.GradScaler()

        self.last_time = time.time()
    
    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        train_datasets = []
        for dataset in cfg.DATASETS.train:
            if dataset.name == 'development':
                dataset_root = path.expanduser(dataset.root)
                dev_dataset = ImageDataset(dataset_root, subset=dataset.subset)
                train_datasets.extend([dev_dataset] * dataset.num_repeat)

        if len(train_datasets) > 1:
            train_dataset = ConcatDataset(train_datasets)
        elif len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            self.print_log('No dataset!')
            exit(0)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=self.local_rank, shuffle=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.GPUS,
            num_workers=cfg.TRAIN.NUM_WORKERS, 
            worker_init_fn=worker_init_fn,
            drop_last=True, pin_memory=True, 
            sampler=self.train_sampler
        )

        val_datasets = []
        for dataset in cfg.DATASETS.val:
            if dataset.name == 'development':
                dataset_root = path.expanduser(dataset.root)
                dev_dataset = ImageDataset(dataset_root, subset=dataset.subset)
                val_datasets.extend([dev_dataset] * dataset.num_repeat)

        if len(val_datasets) > 1:
            val_dataset = ConcatDataset(val_datasets)
        elif len(val_datasets) == 1:
            val_dataset = val_datasets[0]
        else:
            self.print_log('No dataset!')
            exit(0)

        # self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=self.local_rank, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=cfg.TRAIN.NUM_WORKERS, 
            # worker_init_fn=worker_init_fn,
            drop_last=True, pin_memory=True, 
            # sampler=self.train_sampler
        )

        self.print_log('Done!')

    def process_pretrained_model(self):
        cfg = self.cfg
        self.pretrain_step = 0
        self.step = cfg.TRAIN.START_STEP
        self.epoch = 0

        if cfg.TRAIN.RESUME and cfg.PRETRAINED:
            self.pretrain_step = self.load_model(cfg.PRETRAINED)
            # self.step = self.load_model(cfg.PRETRAINED)
            # self.epoch = int(np.ceil(self.step / len(self.train_loader)))
            self.print_log('Load pretrained VOS model from {}.'.format(cfg.PRETRAINED))
            self.print_log('Resume from step {}'.format(self.step))

        elif cfg.PRETRAINED:
            self.load_network(cfg.PRETRAINED)
            self.print_log('Load pretrained VOS model from {}.'.format(cfg.PRETRAINED))
        
    def training(self):
        # torch.autograd.set_detect_anomaly(True)
        cfg = self.cfg 

        # self.report_interval = cfg.LOG.REPORT_INTERVAL
        self.report_interval = cfg.LOG.REPORT_INTERVAL
        self.save_im_interval = cfg.LOG.SAVE_IM_INTERVAL 
        self.save_model_interval = cfg.LOG.SAVE_MODEL_INTERVAL

        step = self.step 
        train_sampler = self.train_sampler
        train_loader = self.train_loader
        valid_loader = self.val_loader 

        epoch = self.epoch

        np.random.seed(np.random.randint(2**30-1) + self.local_rank*100)
        total_epoch = math.ceil(cfg.TRAIN.TOTAL_STEPS/len(train_loader))
        
        # TRAINING 

        while step < cfg.TRAIN.TOTAL_STEPS:            
            epoch += 1
            last_time = time.time()
            self.print_log(f'Epoch {epoch}')
            
            # Crucial for randomness! 
            train_sampler.set_epoch(epoch)
            # Train loop 
            self.train()
            progressbar = tqdm(train_loader) if self.local_rank == 0 else train_loader
            for data in progressbar:
                    if self.pretrain_step > step:
                        step += 1 
                        continue 
                    # continue 
                    step += 1
                    losses = self.forward_sample(data, step)
        
                    if self.local_rank == 0:
                        progressbar.set_description(f"Step {step}, loss: {losses}")
                    
                    if step >= cfg.TRAIN.TOTAL_STEPS:
                        break
            
            # Validate loop 
            if self.pretrain_step <= step:
                self.val() 
                self.forward_val(valid_loader, step)
            # break 
    
    
    def forward_val(self, valid_loader, step):
        self.last_time = time.time()
        self.integrator.finalize('train', step)
        self.integrator.reset_except_hooks()

        torch.set_grad_enabled(self._is_train)
        with torch.cuda.amp.autocast(enabled=self.cfg.AMP):
            progressbar = tqdm(valid_loader) if self.local_rank == 0 else valid_loader
            for i, data in enumerate(progressbar):
                for k, v in data.items():
                    if type(v) != list and type(v) != dict and type(v) != int:
                        data[k] = v.cuda(non_blocking=True)

                # Feed 
                # print("Hello")
                Fs = data['img'] # B, T, C, H, W 
                Ms = data['mask']
                B, C, H, W = Fs.shape 
                logits, history = self.network(Fs, Ms, 0)
                losses = self.network.module.lossComputer.compute_loss(logits, Ms, step)
                probs = F.softmax(logits, dim = 1) 
                preds = torch.argmax(probs, dim = 1)
      
                metrics = calculate_all_metrics(preds, Ms)
                self.integrator.add_dict(losses)
                self.integrator.add_dict(metrics)
                if i % 10 == 0 and self.logger:
                    self.logger.log_images(Fs, Ms, preds, metrics, step, 'val', calculate_uncertainty(probs))
                
                tag = ''
                for loss in losses: 
                    tag += f'_{loss}:{losses[loss]:2f}'
                tag += ';'
                for metric in metrics: 
                    tag += f'_{metric}:{metrics[metric]:2f}'
                progressbar.set_description(f'Step: {i}, {tag}')
        
            self.integrator.finalize('valid', step)
            self.integrator.reset_except_hooks()

    def forward_sample(self, data, step):
        now_lr = adjust_learning_rate(
            optimizer=self.optimizer, 
            base_lr=self.base_lr, 
            p=0.9, 
            itr=step, 
            max_itr=self.cfg.TRAIN.TOTAL_STEPS, 
            warm_up_steps=self.cfg.OPTIMIZER.WARM_UP_STEPS, 
        )

        torch.set_grad_enabled(self._is_train)
        with torch.cuda.amp.autocast(enabled=self.cfg.AMP):
            for k, v in data.items():
                if type(v) != list and type(v) != dict and type(v) != int:
                    data[k] = v.cuda(non_blocking=True)

            # Feed 
            Fs = data['img'] # B, T, C, H, W 
            Ms = data['mask']
            B, C, H, W = Fs.shape 
            logits, losses = self.network(Fs, Ms, step)
            
            self.integrator.add_dict(losses)

            # Compute metric 
            preds = torch.argmax(F.softmax(F.interpolate(
                    logits, size = (Ms.shape[-2:]), mode="bilinear", align_corners=False
                ), dim = 1), dim = 1)

            metrics = calculate_all_metrics(preds, Ms)
            self.integrator.add_dict(metrics)
            

            if step % self.report_interval == 0:
                if self.logger is not None:
                    self.logger.log_metrics('train','lr', now_lr, step)
                    self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval / self.cfg.TRAIN.BATCH_SIZE, step)
                    self.last_time = time.time()
                    self.integrator.finalize('train', step)
                    self.integrator.reset_except_hooks()

            if step % self.save_model_interval == 0 and step != 0:
                if self.logger is not None:
                    self.save(step)
            
        # Backward pass

        self.optimizer.zero_grad(set_to_none=True)
        if self.cfg.AMP:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)    
            self.scaler.update()
        else:   
            losses['total_loss'].backward() 
            self.optimizer.step()

        if step in self.cfg.SCHEDULER.STEPS:
            self.base_lr *= self.cfg.SCHEDULER.GAMMA
        # self.scheduler.step()
        pass 
        tag = ''
        for loss in losses: 
            tag += f'_{loss}:{losses[loss]:2f}'
        tag += ';'
                
        for metric in metrics: 
            tag += f'_{metric}:{metrics[metric]:2f}'
        return tag

    def print_log(self, st): 
        if self.local_rank == 0: 
            print(st) 

    def save(self, step):
        if self.save_path is None:
            print("Saving has been disabled.")
            return 
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % step)
        torch.save(self.network.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)
        
        lst_ckpt = list(map(str, sorted(Path(os.path.dirname(self.save_path)).iterdir(), key=os.path.getmtime)))
        if len(lst_ckpt) > 10:
          print("Remove checkpoint %s" % lst_ckpt[0])
          os.remove(lst_ckpt[0])

        self.save_checkpoint(step)
    
    def save_checkpoint(self, step):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': step,
            'network': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

    def load_model(self, path):
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        # checkpoint = torch.load(path, strict=False) 

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        # map_location = 'cuda:%d' % self.local_rank
        self.network.module.load_state_dict(network, strict = False)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it
    
    def load_network(self, path):
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.network.module.load_state_dict(src_dict, strict=False)
        print('Network weight loaded:', path)

    def train(self):
        self.print_log("Switching to training mode...")
        # self.flag = 'train'
        self._is_train = True
        self._do_log = True
        # Shall be in eval() mode to freeze BN parameters
        # print("Eval network")
        self.network.train()
        self.network.module.train_()
        return self

    def val(self):
        self.print_log("Switching to validating mode...")
        self._is_train = False
        self._do_log = True
        self.network.eval()
        self.network.module.eval_() 
        return self


