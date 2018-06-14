# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Functions for evaluating results on Cityscapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import logging
import os
import uuid

import pycocotools.mask as mask_util

from detectron.core.config import cfg
from detectron.datasets.dataset_catalog import get_raw_dir

logger = logging.getLogger(__name__)


def evaluate_masks(
    json_dataset,
    all_boxes,
    all_segms,
    output_dir,
    use_salt=True,
    cleanup=False
):
    if cfg.CLUSTER.ON_CLUSTER:
        # On the cluster avoid saving these files in the job directory
        output_dir = '/tmp'
    res_file = os.path.join(
        output_dir, 'segmentations_' + json_dataset.name + '_results')
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'

    results_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    os.environ['CITYSCAPES_DATASET'] = get_raw_dir(json_dataset.name)
    os.environ['CITYSCAPES_RESULTS'] = output_dir

    # Load the Cityscapes eval script *after* setting the required env vars,
    # since the script reads their values into global variables (at load time).
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling \
        as cityscapes_eval

    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        im_name = entry['image']

        basename = os.path.splitext(os.path.basename(im_name))[0]
        txtname = os.path.join(output_dir, basename + 'pred.txt')
        with open(txtname, 'w') as fid_txt:
            if i % 10 == 0:
                logger.info('i: {}: {}'.format(i, basename))
            for j in range(1, len(all_segms)):
                clss = json_dataset.classes[j]
                clss_id = cityscapes_eval.name2label[clss].id
                segms = all_segms[j][i]
                boxes = all_boxes[j][i]
                if segms == []:
                    continue
                masks = mask_util.decode(segms)

                for k in range(boxes.shape[0]):
                    score = boxes[k, -1]
                    mask = masks[:, :, k]
                    pngname = os.path.join(
                        'results',
                        basename + '_' + clss + '_{}.png'.format(k))
                    # write txt
                    fid_txt.write('{} {} {}\n'.format(pngname, clss_id, score))
                    # save mask
                    cv2.imwrite(os.path.join(output_dir, pngname), mask * 255)
    logger.info('Evaluating...')
    cityscapes_eval.main([])
    return None
