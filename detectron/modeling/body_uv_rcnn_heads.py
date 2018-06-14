# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core

from detectron.core.config import cfg

import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils

# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #

def add_body_uv_outputs(model, blob_in, dim, pref=''):
    ####
    model.ConvTranspose(blob_in, 'AnnIndex_lowres'+pref, dim, 15,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(blob_in, 'Index_UV_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(
        blob_in, 'U_lowres'+pref, dim, (cfg.BODY_UV_RCNN.NUM_PATCHES+1),
        cfg.BODY_UV_RCNN.DECONV_KERNEL,
        pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
        stride=2,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.}))
    #####
    model.ConvTranspose(
            blob_in, 'V_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,
            cfg.BODY_UV_RCNN.DECONV_KERNEL,
            pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    ####
    blob_Ann_Index = model.BilinearInterpolation('AnnIndex_lowres'+pref, 'AnnIndex'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_Index = model.BilinearInterpolation('Index_UV_lowres'+pref, 'Index_UV'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_U = model.BilinearInterpolation('U_lowres'+pref, 'U_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_V = model.BilinearInterpolation('V_lowres'+pref, 'V_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    ###
    return blob_U,blob_V,blob_Index,blob_Ann_Index


def add_body_uv_losses(model, pref=''):

    ## Reshape for GT blobs.
    model.net.Reshape( ['body_uv_X_points'], ['X_points_reshaped'+pref, 'X_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_Y_points'], ['Y_points_reshaped'+pref, 'Y_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_I_points'], ['I_points_reshaped'+pref, 'I_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_Ind_points'], ['Ind_points_reshaped'+pref, 'Ind_points_shape'+pref],  shape=( -1 ,1 ) )
    ## Concat Ind,x,y to get Coordinates blob.
    model.net.Concat( ['Ind_points_reshaped'+pref,'X_points_reshaped'+pref, \
                       'Y_points_reshaped'+pref],['Coordinates'+pref,'Coordinate_Shapes'+pref ], axis = 1 )
    ##
    ### Now reshape UV blobs, such that they are 1x1x(196*NumSamples)xNUM_PATCHES 
    ## U blob to
    ##
    model.net.Reshape(['body_uv_U_points'], \
                      ['U_points_reshaped'+pref, 'U_points_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['U_points_reshaped'+pref] ,['U_points_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['U_points_reshaped_transpose'+pref], \
                      ['U_points'+pref, 'U_points_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ## V blob
    ##
    model.net.Reshape(['body_uv_V_points'], \
                      ['V_points_reshaped'+pref, 'V_points_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['V_points_reshaped'+pref] ,['V_points_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['V_points_reshaped_transpose'+pref], \
                      ['V_points'+pref, 'V_points_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ###
    ## UV weights blob
    ##
    model.net.Reshape(['body_uv_point_weights'], \
                      ['Uv_point_weights_reshaped'+pref, 'Uv_point_weights_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['Uv_point_weights_reshaped'+pref] ,['Uv_point_weights_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['Uv_point_weights_reshaped_transpose'+pref], \
                      ['Uv_point_weights'+pref, 'Uv_point_weights_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))

    #####################
    ###  Pool IUV for points via bilinear interpolation.
    model.PoolPointsInterp(['U_estimated','Coordinates'+pref], ['interp_U'+pref])
    model.PoolPointsInterp(['V_estimated','Coordinates'+pref], ['interp_V'+pref])
    model.PoolPointsInterp(['Index_UV'+pref,'Coordinates'+pref], ['interp_Index_UV'+pref])

    ## Reshape interpolated UV coordinates to apply the loss.
    
    model.net.Reshape(['interp_U'+pref], \
                      ['interp_U_reshaped'+pref, 'interp_U_shape'+pref],\
                      shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    
    model.net.Reshape(['interp_V'+pref], \
                      ['interp_V_reshaped'+pref, 'interp_V_shape'+pref],\
                      shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ###

    ### Do the actual labels here !!!!
    model.net.Reshape( ['body_uv_ann_labels'],    \
                      ['body_uv_ann_labels_reshaped'   +pref, 'body_uv_ann_labels_old_shape'+pref], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    
    model.net.Reshape( ['body_uv_ann_weights'],   \
                      ['body_uv_ann_weights_reshaped'   +pref, 'body_uv_ann_weights_old_shape'+pref], \
                      shape=( -1 , cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    ###
    model.net.Cast( ['I_points_reshaped'+pref], ['I_points_reshaped_int'+pref], to=core.DataType.INT32)
    ### Now add the actual losses 
    ## The mask segmentation loss (dense)
    probs_seg_AnnIndex, loss_seg_AnnIndex = model.net.SpatialSoftmaxWithLoss( \
                          ['AnnIndex'+pref, 'body_uv_ann_labels_reshaped'+pref,'body_uv_ann_weights_reshaped'+pref],\
                          ['probs_seg_AnnIndex'+pref,'loss_seg_AnnIndex'+pref], \
                           scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    ## Point Patch Index Loss.
    probs_IndexUVPoints, loss_IndexUVPoints = model.net.SoftmaxWithLoss(\
                          ['interp_Index_UV'+pref,'I_points_reshaped_int'+pref],\
                          ['probs_IndexUVPoints'+pref,'loss_IndexUVPoints'+pref], \
                          scale=cfg.BODY_UV_RCNN.PART_WEIGHTS / cfg.NUM_GPUS, spatial=0)
    ## U and V point losses.
    loss_Upoints = model.net.SmoothL1Loss( \
                          ['interp_U_reshaped'+pref, 'U_points'+pref, \
                               'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                          'loss_Upoints'+pref, \
                            scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS  / cfg.NUM_GPUS)
    
    loss_Vpoints = model.net.SmoothL1Loss( \
                          ['interp_V_reshaped'+pref, 'V_points'+pref, \
                               'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                          'loss_Vpoints'+pref, scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)
    ## Add the losses.
    loss_gradients = blob_utils.get_loss_gradients(model, \
                       [ loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints])
    model.losses = list(set(model.losses + \
                       ['loss_Upoints'+pref , 'loss_Vpoints'+pref , \
                        'loss_seg_AnnIndex'+pref ,'loss_IndexUVPoints'+pref]))

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #

def add_ResNet_roi_conv5_head_for_bodyUV(
        model, blob_in, dim_in, spatial_scale
):
    """Add a ResNet "conv5" / "stage5" head for body UV prediction."""
    model.RoIFeatureTransform(
        blob_in, '_[body_uv]_pool5',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
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
    """v1convX design: X * (conv)."""
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
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim

    return current, hidden_dim
