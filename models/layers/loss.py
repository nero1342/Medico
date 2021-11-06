import torch 
import torch.nn.functional as F 
import torch.nn as nn 

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, output, target, it):
        if it < self.start_warm:
            loss = F.cross_entropy(output, target)
            return loss, 1.0, loss

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
        self.dice_weight = [0.25, 0.75]
        self.bce = BootstrappedCE(**cfg.LOSS.data)
        self.dice = DiceLoss(weight = torch.tensor([0.25, 0.75])) 

    def compute_loss(self, predictions, targets, step):
        predictions = F.interpolate(
            predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False
        )
        loss, p, raw_loss = self.bce(predictions, targets, step)

        dice = self.dice(predictions, F.one_hot(targets, 2).permute(0, 3, 1, 2)) 

        losses = {
            "loss_dice": dice * self.loss_weight,
            "loss_sem_seg": loss * self.loss_weight,
            "p": p,
            "raw_loss_seg": raw_loss
            }

        losses['total_loss'] = losses['loss_sem_seg']  + losses["loss_dice"] 
        return losses

    def compute_point_loss(self, point_logits, point_targets, losses = None, step = 0):
        loss = F.cross_entropy(point_logits, point_targets)
        if losses is None:
            losses = {}

        losses["loss_sem_seg_point"] = loss * self.point_loss_weight
        # print(losses['loss_sem_seg'] , losses['loss_sem_seg_point'])
        losses['total_loss'] = losses['total_loss'] + losses['loss_sem_seg_point']
        return losses 