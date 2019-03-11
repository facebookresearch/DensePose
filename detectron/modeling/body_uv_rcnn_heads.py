# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Various network "heads" for dense human pose estimation in DensePose.

The design is as follows:

... -> RoI ----\                                   /-> mask output  -> cls loss
                -> RoIFeatureXform -> body UV head  -> patch output -> cls loss
... -> Feature /                                   \-> UV output    -> reg loss
       Map

The body UV head produces a feature representation of the RoI for the purpose
of dense semantic mask prediction, body surface patch prediction and body UV
coordinates regression. The body UV output module converts the feature
representation into heatmaps for dense mask, patch index and UV coordinates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils

# ---------------------------------------------------------------------------- #
# Body UV outputs and losses
# ---------------------------------------------------------------------------- #

def add_body_uv_outputs(model, blob_in, dim):
    """Add DensePose body UV specific outputs: heatmaps of dense mask, patch index
    and patch-specific UV coordinates. All dense masks are mapped to labels in
    [0, ... S] for S semantically meaningful body parts.
    """
    # Apply ConvTranspose to the feature representation; results in 2x upsampling
    for name in ['AnnIndex', 'Index_UV', 'U', 'V']:
        if name == 'AnnIndex':
            dim_out = cfg.BODY_UV_RCNN.NUM_SEMANTIC_PARTS + 1
        else:
            dim_out = cfg.BODY_UV_RCNN.NUM_PATCHES + 1
        model.ConvTranspose(
            blob_in,
            name + '_lowres',
            dim,
            dim_out,
            cfg.BODY_UV_RCNN.DECONV_KERNEL,
            pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    # Increase heatmap output size via bilinear upsampling
    blob_outputs = []
    for name in ['AnnIndex', 'Index_UV', 'U', 'V']:
        blob_outputs.append(
            model.BilinearInterpolation(
                name + '_lowres',
                name + '_estimated' if name in ['U', 'V'] else name,
                cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
                cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
                cfg.BODY_UV_RCNN.UP_SCALE
            )
        )

    return blob_outputs


def add_body_uv_losses(model):
    """Add DensePose body UV specific losses."""
    # Pool estimated IUV points via bilinear interpolation.
    for name in ['U', 'V', 'Index_UV']:
        model.PoolPointsInterp(
            [
                name + '_estimated' if name in ['U', 'V'] else name,
                'body_uv_coords_xy'
            ],
            ['interp_' + name]
        )
    
    # Compute spatial softmax normalized probabilities, after which
    # cross-entropy loss is computed for semantic parts classification.
    probs_AnnIndex, loss_AnnIndex = model.net.SpatialSoftmaxWithLoss(
        [
            'AnnIndex', 
            'body_uv_parts', 'body_uv_parts_weights'
        ],
        ['probs_AnnIndex', 'loss_AnnIndex'],
        scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS
    )
    # Softmax loss for surface patch classification.
    probs_I_points, loss_I_points = model.net.SoftmaxWithLoss(
        ['interp_Index_UV', 'body_uv_I_points'],
        ['probs_I_points', 'loss_I_points'],
        scale=cfg.BODY_UV_RCNN.PART_WEIGHTS / cfg.NUM_GPUS, 
        spatial=0
    )
    ## Smooth L1 loss for each patch-specific UV coordinates regression.
    # Reshape U,V blobs of both interpolated and ground-truth to compute
    # summarized (instead of averaged) SmoothL1Loss.
    loss_UV = list()
    model.net.Reshape(
        ['body_uv_point_weights'],
        ['UV_point_weights', 'body_uv_point_weights_shape'],
        shape=(1, -1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1)
    )
    for name in ['U', 'V']:
        # Reshape U/V coordinates of both interpolated points and ground-truth
        # points from (#points, #patches) to (1, #points, #patches).
        model.net.Reshape(
            ['body_uv_' + name + '_points'],
            [name + '_points', 'body_uv_' + name + '_points_shape'],
            shape=(1, -1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1)
        )
        model.net.Reshape(
            ['interp_' + name],
            ['interp_' + name + '_reshaped', 'interp_' + name + 'shape'],
            shape=(1, -1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1)
        )
        # Compute summarized SmoothL1Loss of all points.
        loss_UV.append(
            model.net.SmoothL1Loss(
                [
                    'interp_' + name + '_reshaped', name + '_points',
                    'UV_point_weights', 'UV_point_weights'
                ],
                'loss_' + name + '_points',
                scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS
            )
        )
    # Add all losses to compute gradients
    loss_gradients = blob_utils.get_loss_gradients(
        model, [loss_AnnIndex, loss_I_points] + loss_UV
    )
    # Update model training losses
    model.AddLosses(
        ['loss_' + name for name in ['AnnIndex', 'I_points', 'U_points', 'V_points']]
    )

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #

def add_ResNet_roi_conv5_head_for_bodyUV(model, blob_in, dim_in, spatial_scale):
    """Add a ResNet "conv5" / "stage5" head for body UV prediction."""
    model.RoIFeatureTransform(
        blob_in, 
        '_[body_uv]_pool5',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    # Using the prefix '_[body_uv]_' to 'res5' enables initializing the head's
    # parameters using pretrained 'res5' parameters if given (see
    # utils.net.initialize_from_weights_file)
    s, dim_in = ResNet.add_stage(
        model,
        '_[body_uv]_res5',
        '_[body_uv]_pool5',
        3,
        dim_in,
        2048,
        512,
        cfg.BODY_UV_RCNN.DILATION,
        stride_init=int(cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION / 7)
    )
    return s, 2048


def add_roi_body_uv_head_v1convX(model, blob_in, dim_in, spatial_scale):
    """Add a DensePose body UV head. v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 1),
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=const_fill(0.0)
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim

    return current, hidden_dim
