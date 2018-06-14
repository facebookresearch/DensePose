# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""VGG_CNN_M_1024 from https://arxiv.org/abs/1405.3531."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg


def add_VGG_CNN_M_1024_conv5_body(model):
    model.Conv('data', 'conv1', 3, 96, 7, pad=0, stride=2)
    model.Relu('conv1', 'conv1')
    model.LRN('conv1', 'norm1', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm1', 'pool1', kernel=3, pad=0, stride=2)
    model.StopGradient('pool1', 'pool1')
    # No updates at conv1 and below (norm1 and pool1 have no params,
    # so we can stop gradients before them, too)
    model.Conv('pool1', 'conv2', 96, 256, 5, pad=0, stride=2)
    model.Relu('conv2', 'conv2')
    model.LRN('conv2', 'norm2', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm2', 'pool2', kernel=3, pad=0, stride=2)
    model.Conv('pool2', 'conv3', 256, 512, 3, pad=1, stride=1)
    model.Relu('conv3', 'conv3')
    model.Conv('conv3', 'conv4', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4', 'conv4')
    model.Conv('conv4', 'conv5', 512, 512, 3, pad=1, stride=1)
    blob_out = model.Relu('conv5', 'conv5')
    return blob_out, 512, 1. / 16.


def add_VGG_CNN_M_1024_roi_fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', 4096, 1024)
    blob_out = model.Relu('fc7', 'fc7')
    return blob_out, 1024
