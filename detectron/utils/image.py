# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Image helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np


def aspect_ratio_rel(im, aspect_ratio):
    """Performs width-relative aspect ratio transformation."""
    im_h, im_w = im.shape[:2]
    im_ar_w = int(round(aspect_ratio * im_w))
    im_ar = cv2.resize(im, dsize=(im_ar_w, im_h))
    return im_ar


def aspect_ratio_abs(im, aspect_ratio):
    """Performs absolute aspect ratio transformation."""
    im_h, im_w = im.shape[:2]
    im_area = im_h * im_w

    im_ar_w = np.sqrt(im_area * aspect_ratio)
    im_ar_h = np.sqrt(im_area / aspect_ratio)
    assert np.isclose(im_ar_w / im_ar_h, aspect_ratio)

    im_ar = cv2.resize(im, dsize=(int(im_ar_w), int(im_ar_h)))
    return im_ar
