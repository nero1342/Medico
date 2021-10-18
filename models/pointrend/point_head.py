# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, cfg, in_channels):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardPointHead, self).__init__()
        # fmt: off
        num_classes                 = cfg.MODEL.POINT_HEAD.NUM_CLASSES
        fc_dim                      = cfg.MODEL.POINT_HEAD.FC_DIM
        num_fc                      = cfg.MODEL.POINT_HEAD.NUM_FC
        cls_agnostic_mask           = cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK
        self.coarse_pred_each_layer = cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER
        input_channels              = in_channels 

        # fmt: on

        fc_dim_in = input_channels + num_classes
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_features), dim=1)
        return self.predictor(x)
