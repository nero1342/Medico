import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from models.pointrend.point_head import StandardPointHead
from models.pointrend.point_features import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from models.layers.loss import LossComputer
from models.deeplab.deeplab import DeepLab
from models.deeplab.network import Network
from utils.uncertainty import calculate_uncertainty
class PointRend(nn.Module):
    """
    A semantic segmentation head that combines a head set in `POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME`
    and a point head set in `MODEL.POINT_HEAD.NAME`.
    """

    def __init__(self, cfg):
        super().__init__()

        self.coarse_sem_seg_head = Network()
        self._init_point_head(cfg)

    def _init_point_head(self, cfg):
        # fmt: off
        # assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        in_channels                  = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.train_num_points        = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.oversample_ratio        = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        self.subdivision_steps       = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.subdivision_num_points  = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # fmt: on

        # in_channels = int(np.sum([feature_channels[f] for f in self.in_features]))
        self.point_head = StandardPointHead(cfg, in_channels)

        self.lossComputer = LossComputer(cfg)

        self.train_() 

    def train_(self):
        self.training = True 
    def eval_(self):
        self.training = False 

    def forward(self, features, targets=None, step = 0):
        coarse_sem_seg_logits, low_level_feat = self.coarse_sem_seg_head(features)

        if self.training:
            losses = self.lossComputer.compute_loss(coarse_sem_seg_logits, targets, step)
    
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    coarse_sem_seg_logits,
                    calculate_uncertainty,
                    self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )

            coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
            fine_grained_features = point_sample(low_level_feat, point_coords, align_corners=False)
            
            point_logits = self.point_head(fine_grained_features, coarse_features)
            point_targets = (
                point_sample(
                    targets.unsqueeze(1).to(torch.float),
                    point_coords,
                    mode="nearest",
                    align_corners=False,
                )
                .squeeze(1)
                .to(torch.long)
            )
            losses = self.lossComputer.compute_point_loss(point_logits, point_targets, losses, step)
            
            return coarse_sem_seg_logits, losses
        else:
            sem_seg_logits = coarse_sem_seg_logits.clone()
            history = [(0, sem_seg_logits, calculate_uncertainty(sem_seg_logits), torch.rand(sem_seg_logits.size(0), 1, 2))]

            for i in range(self.subdivision_steps):
                sem_seg_logits = F.interpolate(
                    sem_seg_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                uncertainty_map = calculate_uncertainty(sem_seg_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.subdivision_num_points
                )
                
                fine_grained_features = point_sample(low_level_feat, point_coords, align_corners=False)
                coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)

                point_logits = self.point_head(fine_grained_features, coarse_features)
                # put sem seg point predictions to the right places on the upsampled grid.
                N, C, H, W = sem_seg_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                sem_seg_logits = (
                    sem_seg_logits.reshape(N, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(N, C, H, W)
                )
                history.append((i + 1, sem_seg_logits, uncertainty_map, point_coords))

            return sem_seg_logits, history