# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()


# Extension modules
ext_modules = [
    Extension(
        name='detectron.utils.cython_bbox',
        sources=[
            'detectron/utils/cython_bbox.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    ),
    Extension(
        name='detectron.utils.cython_nms',
        sources=[
            'detectron/utils/cython_nms.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    )
]

setup(
    name='Detectron',
    packages=['detectron'],
    ext_modules=cythonize(ext_modules)
)
