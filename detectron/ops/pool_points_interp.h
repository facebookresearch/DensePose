/**
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 */


#ifndef POOL_POINTS_INTERP_OP_H_
#define POOL_POINTS_INTERP_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class PoolPointsInterpOp final : public Operator<Context> {
 public:
  PoolPointsInterpOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        spatial_scale_(OperatorBase::GetSingleArgument<float>(
              "spatial_scale", 1.)) {
    DCHECK_GT(spatial_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
};

template <typename T, class Context>
class PoolPointsInterpGradientOp final : public Operator<Context> {
 public:
  PoolPointsInterpGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(OperatorBase::GetSingleArgument<float>(
              "spatial_scale", 1.)){
    DCHECK_GT(spatial_scale_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
};

} // namespace caffe2

#endif // PoolPointsInterpOp
