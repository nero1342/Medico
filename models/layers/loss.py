import torch 
import torch.nn.functional as F 
import torch.nn as nn 

# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, output, target, it):
        if it < self.start_warm:
            return F.cross_entropy(output, target), 1.0, -1

        raw_loss = F.cross_entropy(output, target, reduction = 'none').view(-1)
        num_pixels = raw_loss.numel()
            
        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p, raw_loss.mean().float()


class LossComputer():
    def __init__(self, cfg):
        pass 
        self.loss_weight = 1
        self.point_loss_weight = 1
        self.bce = BootstrappedCE(**cfg.LOSS.data)

    def compute_loss(self, predictions, targets, step):
        predictions = F.interpolate(
            predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False
        )
        loss, p, raw_loss = self.bce(predictions, targets, step)
        losses = {
            "loss_sem_seg": loss * self.loss_weight,
            "p": p,
            "raw_loss_seg": raw_loss
            }

        losses['total_loss'] = losses['loss_sem_seg']
        return losses

    def compute_point_loss(self, point_logits, point_targets, losses = None, step = 0):
        loss = F.cross_entropy(point_logits, point_targets)
        if losses is None:
            losses = {}

        losses["loss_sem_seg_point"] = loss * self.point_loss_weight
        # print(losses['loss_sem_seg'] , losses['loss_sem_seg_point'])
        losses['total_loss'] = losses['loss_sem_seg'] + losses['loss_sem_seg_point']
        return losses 