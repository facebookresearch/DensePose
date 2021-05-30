/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pool_points_interp.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(PoolPointsInterp,
                      PoolPointsInterpOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(PoolPointsInterpGradient,
                      PoolPointsInterpGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(PoolPointsInterp)
    .NumInputs(2)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "4D feature/heat map input of shape (N, C, H, W).")
    .Input(
        1,
        "coords",
        "2D input of shape (P, 2) specifying P points with 2 columns "
        "representing 2D coordinates on the image (x, y). The "
        "coordinates have been converted to in the coordinate system of X.")
    .Output(
        0,
        "Y",
        "2D output of shape (P, K). The r-th batch element is a "
        "pooled/interpolated index or UV coordinate corresponding "
        "to the r-th point over all K patches (including background).");

OPERATOR_SCHEMA(PoolPointsInterpGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "See PoolPointsInterp.")
    .Input(
        1,
        "coords",
        "See PoolPointsInterp.")
    .Input(
        2,
        "dY",
        "Gradient of forward output 0 (Y)")
    .Output(
        0,
        "dX",
        "Gradient of forward input 0 (X)");

class GetPoolPointsInterpGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PoolPointsInterpGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(PoolPointsInterp, GetPoolPointsInterpGradient);

} // namespace caffe2
