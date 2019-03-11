# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Construct minibatches for DensePose training. Handles the minibatch blobs
that are specific to DensePose. Other blobs that are generic to RPN or
Fast/er R-CNN are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import logging
import numpy as np

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.segms as segm_utils
import detectron.utils.densepose_methods as dp_utils

logger = logging.getLogger(__name__)

DP = dp_utils.DensePoseMethods()


def add_body_uv_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add DensePose specific blobs to the given inputs blobs dictionary."""
    M = cfg.BODY_UV_RCNN.HEATMAP_SIZE
    # Prepare the body UV targets by associating one gt box which contains
    # body UV annotations to each training roi that has a fg class label.
    polys_gt_inds = np.where(roidb['ignore_UV_body'] == 0)[0]
    boxes_from_polys = roidb['boxes'][polys_gt_inds]
    # Select foreground RoIs
    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_body_uv = np.zeros_like(blobs['labels_int32'], dtype=np.int32)

    if ((boxes_from_polys.shape[0] > 0) & (fg_inds.shape[0] > 0)):
        # Find overlap between all foreground RoIs and the gt bounding boxes
        # containing each body UV annotaion.
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False)
        )
        # Select foreground RoIs as those with > 0.7 overlap
        fg_polys_value = np.max(overlaps_bbfg_bbpolys, axis=1)
        fg_inds = fg_inds[fg_polys_value > 0.7]

    if ((boxes_from_polys.shape[0] > 0) & (fg_inds.shape[0] > 0)):
        roi_has_body_uv[fg_inds] = 1
        # Create body UV blobs
        # Dense masks, each mask for a given fg roi is of size M x M.
        part_inds = blob_utils.zeros((fg_inds.shape[0], M, M), int32=True)
        # Weights assigned to each target in `part_inds`. By default, all 1's.
        # part_inds_weights = blob_utils.zeros((fg_inds.shape[0], M, M), int32=True)
        part_inds_weights = blob_utils.ones((fg_inds.shape[0], M, M), int32=False)
        # 2D spatial coordinates (on the image). Shape is (#fg_rois, 2) in format
        # (x, y).
        coords_xy = blob_utils.zeros((fg_inds.shape[0], 196, 2), int32=False)
        # 24 patch indices plus a background class
        I_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=True)
        # UV coordinates in each patch
        U_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        V_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        # Uv_point_weights = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)

        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = overlaps_bbfg_bbpolys[fg_inds]
        # Map from each fg roi to the index of the gt box with highest overlap
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # Add body UV targets for each fg roi
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            polys_gt_ind = polys_gt_inds[fg_polys_ind]
            # RLE encoded dense masks which are of size 256 x 256.
            # Map all part masks to 14 labels (i.e., indices of semantic body parts).
            dp_masks = dp_utils.GetDensePoseMask(
                roidb['dp_masks'][polys_gt_ind], cfg.BODY_UV_RCNN.NUM_SEMANTIC_PARTS
            )
            # Surface patch indices of collected points
            dp_I = np.array(roidb['dp_I'][polys_gt_ind], dtype=np.int32)
            # UV coordinates of collected points
            dp_U = np.array(roidb['dp_U'][polys_gt_ind], dtype=np.float32)
            dp_V = np.array(roidb['dp_V'][polys_gt_ind], dtype=np.float32)
            # dp_UV_weights = np.ones_like(dp_I).astype(np.float32)
            # Spatial coordinates on the image which are scaled such that the bbox
            # size is 256 x 256.
            dp_x = np.array(roidb['dp_x'][polys_gt_ind], dtype=np.float32)
            dp_y = np.array(roidb['dp_y'][polys_gt_ind], dtype=np.float32)
            # Do the flipping of the densepose annotation
            if roidb['flipped']:
                dp_I, dp_U, dp_V, dp_x, dp_y, dp_masks = DP.get_symmetric_densepose(
                    dp_I, dp_U, dp_V, dp_x, dp_y, dp_masks
                )

            roi_fg = rois_fg[i]
            gt_box = boxes_from_polys[fg_polys_ind]
            fg_x1, fg_y1, fg_x2, fg_y2 = roi_fg[0:4]
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box[0:4]
            fg_width = fg_x2 - fg_x1;  fg_height = fg_y2 - fg_y1
            gt_width = gt_x2 - gt_x1;  gt_height = gt_y2 - gt_y1
            fg_scale_w = float(M) / fg_width
            fg_scale_h = float(M) / fg_height
            gt_scale_w = 256. / gt_width
            gt_scale_h = 256. / gt_height
            # Sample M points evenly within the fg roi and scale the relative coordinates
            # (to associated gt box) such that the bounding box size is 256 x 256.
            x_targets = (np.arange(fg_x1, fg_x2, fg_width / M) - gt_x1) * gt_scale_w
            y_targets = (np.arange(fg_y1, fg_y2, fg_height / M) - gt_y1) * gt_scale_h
            # Construct 2D coordiante matrices
            x_targets, y_targets = np.meshgrid(x_targets[:M], y_targets[:M])
            ## Another implementation option (which results in similar performance)
            # x_targets = (np.linspace(fg_x1, fg_x2, M, endpoint=True, dtype=np.float32) - gt_x1) * gt_scale_w
            # y_targets = (np.linspace(fg_y1, fg_y2, M, endpoint=True, dtype=np.float32) - gt_y1) * gt_scale_h
            # x_targets = (np.linspace(fg_x1, fg_x2, M, endpoint=False) - gt_x1) * gt_scale_w
            # y_targets = (np.linspace(fg_y1, fg_y2, M, endpoint=False) - gt_y1) * gt_scale_h
            # x_targets, y_targets = np.meshgrid(x_targets, y_targets)

            # Map dense masks of size 256 x 256 to target heatmap of size M x M.
            part_inds[i] = cv2.remap(
                dp_masks, x_targets.astype(np.float32), y_targets.astype(np.float32),
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0)
            )

            # Scale annotated spatial coordinates from bbox of size 256 x 256 to target
            # heatmap of size M x M.
            dp_x = (dp_x / gt_scale_w + gt_x1 - fg_x1) * fg_scale_w
            dp_y = (dp_y / gt_scale_h + gt_y1 - fg_y1) * fg_scale_h   
            # Set patch index of points outside the heatmap as 0 (background).
            dp_I[dp_x < 0] = 0; dp_I[dp_x > (M - 1)] = 0
            dp_I[dp_y < 0] = 0; dp_I[dp_y > (M - 1)] = 0
            # Get body UV annotations of points inside the heatmap.
            points_inside = dp_I > 0
            dp_x = dp_x[points_inside]
            dp_y = dp_y[points_inside]
            dp_I = dp_I[points_inside]
            dp_U = dp_U[points_inside]
            dp_V = dp_V[points_inside] 
            # dp_UV_weights = dp_UV_weights[points_inside]

            # Update body UV blobs
            num_dp_points = len(dp_I)
            # coords_xy[i, 0:num_dp_points, 0] = i  # fg_roi index
            coords_xy[i, 0:num_dp_points, 0] = dp_x
            coords_xy[i, 0:num_dp_points, 1] = dp_y
            I_points[i, 0:num_dp_points] = dp_I.astype(np.int32)
            U_points[i, 0:num_dp_points] = dp_U
            V_points[i, 0:num_dp_points] = dp_V
            # Uv_point_weights[i, 0:len(dp_UV_weights)] = dp_UV_weights
    else:  # If there are no fg rois
        # The network cannot handle empty blobs, so we must provide a blob.
        # We simply take the first bg roi, give it an all 0's body UV annotations
        # and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # `rois_fg` is actually one background roi, but that's ok because ...
        if len(bg_inds) == 0:
            rois_fg = sampled_boxes[0].reshape((1, -1))
        else:
            rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # Mark that the first roi has body UV annotation
        roi_has_body_uv[0] = 1
        # We give it all 0's blobs
        part_inds = blob_utils.zeros((1, M, M), int32=True)
        part_inds_weights = blob_utils.zeros((1, M, M), int32=False)
        coords_xy = blob_utils.zeros((1, 196, 2), int32=False)
        I_points   = blob_utils.zeros((1, 196), int32=True)
        U_points   = blob_utils.zeros((1, 196), int32=False)
        V_points   = blob_utils.zeros((1, 196), int32=False)
        # Uv_point_weights = blob_utils.zeros((1, 196), int32=False)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))
    # Create body UV blobs for all patches (including background)
    K = cfg.BODY_UV_RCNN.NUM_PATCHES + 1
    # Construct U/V_points blobs for all patches by repeating it #num_patches times.
    # Shape: (#rois, 196, K)
    U_points = np.repeat(U_points[:, :, np.newaxis], K, axis=-1)
    V_points = np.repeat(V_points[:, :, np.newaxis], K, axis=-1)
    uv_point_weights = np.zeros_like(U_points)
    # Set binary weights for UV targets in each patch
    for i in np.arange(1, K):
        uv_point_weights[:, :, i] = (I_points == i).astype(np.float32)

    # Update blobs dict with body UV blobs
    blobs['body_uv_rois'] = rois_fg
    blobs['roi_has_body_uv_int32'] = roi_has_body_uv  # shape: (#rois,)
    blobs['body_uv_parts'] = part_inds  # shape: (#rois, M, M)
    blobs['body_uv_parts_weights'] = part_inds_weights
    blobs['body_uv_coords_xy'] = coords_xy.reshape(-1, 2)  # shape: (#rois * 196, 2)
    blobs['body_uv_I_points'] = I_points.reshape(-1, 1)  # shape: (#rois * 196, 1)
    blobs['body_uv_U_points'] = U_points  # shape: (#rois, 196, K)
    blobs['body_uv_V_points'] = V_points
    blobs['body_uv_point_weights'] = uv_point_weights
