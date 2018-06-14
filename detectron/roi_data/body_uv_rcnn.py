# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#
from scipy.io import loadmat
import copy
import cv2
import logging
import numpy as np
#

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.segms as segm_utils
import detectron.utils.densepose_methods as dp_utils

#
from memory_profiler import profile
#
import os
#
logger = logging.getLogger(__name__)
#
DP = dp_utils.DensePoseMethods()
#

def add_body_uv_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    IsFlipped = roidb['flipped']
    M = cfg.BODY_UV_RCNN.HEATMAP_SIZE
    #
    polys_gt_inds = np.where(roidb['ignore_UV_body'] == 0)[0]
    boxes_from_polys = [roidb['boxes'][i,:] for i in polys_gt_inds]
    if not(boxes_from_polys):
        pass
    else:
        boxes_from_polys = np.vstack(boxes_from_polys)
    boxes_from_polys = np.array(boxes_from_polys)

    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = np.zeros( blobs['labels_int32'].shape )

    if (bool(boxes_from_polys.any()) & (fg_inds.shape[0] > 0) ):
        rois_fg = sampled_boxes[fg_inds]
        #
        rois_fg.astype(np.float32, copy=False)
        boxes_from_polys.astype(np.float32, copy=False)
        #
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False))
        fg_polys_value = np.max(overlaps_bbfg_bbpolys, axis=1)
        fg_inds = fg_inds[fg_polys_value>0.7]

    if (bool(boxes_from_polys.any()) & (fg_inds.shape[0] > 0) ):
        for jj in fg_inds:
            roi_has_mask[jj] = 1
         
        # Create blobs for densepose supervision.
        ################################################## The mask
        All_labels = blob_utils.zeros((fg_inds.shape[0], M ** 2), int32=True)
        All_Weights = blob_utils.zeros((fg_inds.shape[0], M ** 2), int32=True)
        ################################################# The points
        X_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        Y_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        Ind_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=True)
        I_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=True)
        U_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        V_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        Uv_point_weights = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        #################################################

        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False))
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        for i in range(rois_fg.shape[0]):
            #
            fg_polys_ind = polys_gt_inds[ fg_polys_inds[i] ]
            #
            Ilabel = segm_utils.GetDensePoseMask( roidb['dp_masks'][ fg_polys_ind ] )
            #
            GT_I = np.array(roidb['dp_I'][ fg_polys_ind ])
            GT_U = np.array(roidb['dp_U'][ fg_polys_ind ])
            GT_V = np.array(roidb['dp_V'][ fg_polys_ind ])
            GT_x = np.array(roidb['dp_x'][ fg_polys_ind ])
            GT_y = np.array(roidb['dp_y'][ fg_polys_ind ])
            GT_weights = np.ones(GT_I.shape).astype(np.float32)
            #
            ## Do the flipping of the densepose annotation !
            if(IsFlipped):
                GT_I,GT_U,GT_V,GT_x,GT_y,Ilabel = DP.get_symmetric_densepose(GT_I,GT_U,GT_V,GT_x,GT_y,Ilabel)
            #
            roi_fg = rois_fg[i]
            roi_gt = boxes_from_polys[fg_polys_inds[i],:]
            #
            x1 = roi_fg[0]  ;   x2 = roi_fg[2]
            y1 = roi_fg[1]  ;   y2 = roi_fg[3]
            #
            x1_source = roi_gt[0];  x2_source = roi_gt[2]
            y1_source = roi_gt[1];  y2_source = roi_gt[3]
            #
            x_targets  = ( np.arange(x1,x2, (x2 - x1)/M ) - x1_source ) * ( 256. / (x2_source-x1_source) )  
            y_targets  = ( np.arange(y1,y2, (y2 - y1)/M ) - y1_source ) * ( 256. / (y2_source-y1_source) )  
            #
            x_targets = x_targets[0:M] ## Strangely sometimes it can be M+1, so make sure size is OK!
            y_targets = y_targets[0:M]
            #
            [X_targets,Y_targets] = np.meshgrid( x_targets, y_targets )
            New_Index = cv2.remap(Ilabel,X_targets.astype(np.float32), Y_targets.astype(np.float32), interpolation=cv2.INTER_NEAREST, borderMode= cv2.BORDER_CONSTANT, borderValue=(0))
            #
            All_L = np.zeros(New_Index.shape)
            All_W = np.ones(New_Index.shape)
            #
            All_L = New_Index
            #
            gt_length_x = x2_source - x1_source
            gt_length_y = y2_source - y1_source
            #
            GT_y =  ((  GT_y / 256. * gt_length_y  ) + y1_source - y1 ) *  ( M /  ( y2 - y1 ) )
            GT_x =  ((  GT_x / 256. * gt_length_x  ) + x1_source - x1 ) *  ( M /  ( x2 - x1 ) )
            #
            GT_I[GT_y<0] = 0
            GT_I[GT_y>(M-1)] = 0
            GT_I[GT_x<0] = 0
            GT_I[GT_x>(M-1)] = 0
            #
            points_inside = GT_I>0
            GT_U = GT_U[points_inside]
            GT_V = GT_V[points_inside]
            GT_x = GT_x[points_inside]
            GT_y = GT_y[points_inside]
            GT_weights = GT_weights[points_inside]
            GT_I = GT_I[points_inside]
            #
            X_points[i, 0:len(GT_x)] = GT_x
            Y_points[i, 0:len(GT_y)] = GT_y
            Ind_points[i, 0:len(GT_I)] = i
            I_points[i, 0:len(GT_I)] = GT_I
            U_points[i, 0:len(GT_U)] = GT_U
            V_points[i, 0:len(GT_V)] = GT_V
            Uv_point_weights[i, 0:len(GT_weights)] = GT_weights
            #
            All_labels[i, :] = np.reshape(All_L.astype(np.int32), M ** 2)
            All_Weights[i, :] = np.reshape(All_W.astype(np.int32), M ** 2)
            ##
    else:
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        #
        if(len(bg_inds)==0):
            rois_fg = sampled_boxes[0].reshape((1, -1))
        else:
            rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))

        roi_has_mask[0] = 1
        #
        X_points = blob_utils.zeros((1, 196), int32=False)
        Y_points = blob_utils.zeros((1, 196), int32=False)
        Ind_points = blob_utils.zeros((1, 196), int32=True)
        I_points = blob_utils.zeros((1,196), int32=True)
        U_points = blob_utils.zeros((1, 196), int32=False)
        V_points = blob_utils.zeros((1, 196), int32=False)
        Uv_point_weights = blob_utils.zeros((1, 196), int32=False)
        #
        All_labels = -blob_utils.ones((1, M ** 2), int32=True) * 0 ## zeros
        All_Weights = -blob_utils.ones((1, M ** 2), int32=True) * 0 ## zeros
    #
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))
    #
    K = cfg.BODY_UV_RCNN.NUM_PATCHES
    #
    U_points = np.tile( U_points , [1,K+1] )
    V_points = np.tile( V_points , [1,K+1] )
    Uv_Weight_Points = np.zeros(U_points.shape)
    #
    for jjj in xrange(1,K+1):
        Uv_Weight_Points[ : , jjj * I_points.shape[1]  : (jjj+1) * I_points.shape[1] ] = ( I_points == jjj ).astype(np.float32)
    #
    ################
    # Update blobs dict with Mask R-CNN blobs
    ###############
    #
    blobs['body_uv_rois'] = np.array(rois_fg)
    blobs['roi_has_body_uv_int32'] = np.array(roi_has_mask).astype(np.int32)
    ##
    blobs['body_uv_ann_labels'] = np.array(All_labels).astype(np.int32)
    blobs['body_uv_ann_weights'] = np.array(All_Weights).astype(np.float32)
    #
    ##########################
    blobs['body_uv_X_points'] = X_points.astype(np.float32)
    blobs['body_uv_Y_points'] = Y_points.astype(np.float32)
    blobs['body_uv_Ind_points'] = Ind_points.astype(np.float32)
    blobs['body_uv_I_points'] = I_points.astype(np.float32)
    blobs['body_uv_U_points'] = U_points.astype(np.float32)  #### VERY IMPORTANT :   These are switched here :
    blobs['body_uv_V_points'] = V_points.astype(np.float32)
    blobs['body_uv_point_weights'] = Uv_Weight_Points.astype(np.float32)
    ###################




