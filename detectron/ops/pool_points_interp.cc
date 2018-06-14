/**
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 */


#include "pool_points_interp.h"

namespace caffe2 {
//namespace {

REGISTER_CPU_OPERATOR(PoolPointsInterp,
                      PoolPointsInterpOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(PoolPointsInterpGradient,
                      PoolPointsInterpGradientOp<float, CPUContext>);

// Input: X, points; Output: Y
OPERATOR_SCHEMA(PoolPointsInterp).NumInputs(2).NumOutputs(1);
// Input: X, points, dY (aka "gradOutput");
// Output: dX (aka "gradInput")
OPERATOR_SCHEMA(PoolPointsInterpGradient).NumInputs(3).NumOutputs(1);

class GetPoolPointsInterpGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "PoolPointsInterpGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(PoolPointsInterp, GetPoolPointsInterpGradient);

//} // namespace
} // namespace caffe2
