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

import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import gradient_checker
from caffe2.python import workspace

import detectron.utils.c2 as c2_utils
import detectron.utils.logging as logging_utils


class SmoothL1LossTest(unittest.TestCase):
    def test_forward_and_gradient(self):
        Y = np.random.randn(128, 4 * 21).astype(np.float32)
        Y_hat = np.random.randn(128, 4 * 21).astype(np.float32)
        inside_weights = np.random.randn(128, 4 * 21).astype(np.float32)
        inside_weights[inside_weights < 0] = 0
        outside_weights = np.random.randn(128, 4 * 21).astype(np.float32)
        outside_weights[outside_weights < 0] = 0
        scale = np.random.random()
        beta = np.random.random()

        op = core.CreateOperator(
            'SmoothL1Loss', ['Y_hat', 'Y', 'inside_weights', 'outside_weights'],
            ['loss'],
            scale=scale,
            beta=beta
        )

        gc = gradient_checker.GradientChecker(
            stepsize=0.005,
            threshold=0.005,
            device_option=core.DeviceOption(caffe2_pb2.CUDA, 0)
        )

        res, grad, grad_estimated = gc.CheckSimple(
            op, [Y_hat, Y, inside_weights, outside_weights], 0, [0]
        )

        self.assertTrue(
            grad.shape == grad_estimated.shape,
            'Fail check: grad.shape != grad_estimated.shape'
        )

        # To inspect the gradient and estimated gradient:
        # np.set_printoptions(precision=3, suppress=True)
        # print('grad:')
        # print(grad)
        # print('grad_estimated:')
        # print(grad_estimated)

        self.assertTrue(res)


if __name__ == '__main__':
    c2_utils.import_detectron_ops()
    assert 'SmoothL1Loss' in workspace.RegisteredOperators()
    logging_utils.setup_logging(__name__)
    unittest.main()
