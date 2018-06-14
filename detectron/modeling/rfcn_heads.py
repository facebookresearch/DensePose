# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill


# ---------------------------------------------------------------------------- #
# R-FCN outputs and losses
# ---------------------------------------------------------------------------- #

def add_rfcn_outputs(model, blob_in, dim_in, dim_reduce, spatial_scale):
    if dim_reduce is not None:
        # Optional dim reduction
        blob_in = model.Conv(
            blob_in,
            'conv_dim_reduce',
            dim_in,
            dim_reduce,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        blob_in = model.Relu(blob_in, blob_in)
        dim_in = dim_reduce
    # Classification conv
    model.Conv(
        blob_in,
        'conv_cls',
        dim_in,
        model.num_classes * cfg.RFCN.PS_GRID_SIZE**2,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    # Bounding-box regression conv
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.Conv(
        blob_in,
        'conv_bbox_pred',
        dim_in,
        4 * num_bbox_reg_classes * cfg.RFCN.PS_GRID_SIZE**2,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    # Classification PS RoI pooling
    model.net.PSRoIPool(
        ['conv_cls', 'rois'], ['psroipooled_cls', '_mapping_channel_cls'],
        group_size=cfg.RFCN.PS_GRID_SIZE,
        output_dim=model.num_classes,
        spatial_scale=spatial_scale
    )
    model.AveragePool(
        'psroipooled_cls', 'cls_score_4d', kernel=cfg.RFCN.PS_GRID_SIZE
    )
    model.net.Reshape(
        'cls_score_4d', ['cls_score', '_cls_scores_shape'],
        shape=(-1, cfg.MODEL.NUM_CLASSES)
    )
    if not model.train:
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    # Bbox regression PS RoI pooling
    model.net.PSRoIPool(
        ['conv_bbox_pred', 'rois'],
        ['psroipooled_bbox', '_mapping_channel_bbox'],
        group_size=cfg.RFCN.PS_GRID_SIZE,
        output_dim=4 * num_bbox_reg_classes,
        spatial_scale=spatial_scale
    )
    model.AveragePool(
        'psroipooled_bbox', 'bbox_pred', kernel=cfg.RFCN.PS_GRID_SIZE
    )
