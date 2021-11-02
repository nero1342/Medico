import wandb
import os 
import torch 
import torch.nn.functional as F 

class WandB():
    def __init__(self, project_name, save_path = "runs", config=None, id = None):
        wandb.init(dir=save_path, project=project_name, config=config, mode = 'online', name=id)
        print("Initial WandB logger")
        print("-- Project Name: ", project_name)
        print("-- Save at: ", save_path + '/' + wandb.run.name)
        print("-- Run ID: ", id)
        
        os.makedirs(save_path, exist_ok = True)
        self.project_name = project_name
        self.id = id 
    
    def watch(self, network):
        wandb.watch(network.module)
    
    def log_scalar(self, tag, value, step = 0):
        wandb.log({tag: value}, step=step)
        
    def log_metrics(self, prefix, tag, value, step = 0):
        text = f'{self.project_name}:{self.id} - It {step} [{prefix.upper()}] [{tag}]: {value:06f}'
        print(text)
        self.log_scalar(f'{prefix.upper()}/{tag}', value, step)

    def log_images(self, Fs, Ms, Es, metrics, step, type = 'train', uncerainty = None):
        # print(Fs.shape, Ms.shape, Es.shape)
        B = Fs.size(0)
        class_labels = {1: "foreground"}
    
        tag = f'{type}/Report_{step // 1000}/{step}'
        for metric in metrics: 
            tag += f'_{metric}:{metrics[metric]:2f}'
        # print(tag)
        imgs = [
            wandb.Image(Fs[i].detach().cpu().permute(1, 2, 0).numpy(),
                        # caption = f"", 
                        masks = {
                "prediction": {
                    "mask_data": Es[i].detach().cpu().numpy(), 
                    "class_labels": class_labels 
                },
                "ground truth": {
                    "mask_data": Ms[i].detach().cpu().numpy(), 
                    "class_labels": class_labels 
                },
                
            })
        for i in range(min(4, B))]
        if uncerainty is not None:
          imgs.extend([
              wandb.Image(uncerainty[i].detach().cpu().numpy())
          for i in range(min(4, B))])
        wandb.log({tag: imgs}, step = step)
            