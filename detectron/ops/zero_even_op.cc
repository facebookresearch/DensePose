/**
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 */

#include "zero_even_op.h"

namespace caffe2 {

template <>
bool ZeroEvenOp<float, CPUContext>::RunOnDevice() {
  // Retrieve the input tensor.
  const auto& X = Input(0);
  CAFFE_ENFORCE(X.ndim() == 1);

  // Initialize the output tensor to a copy of the input tensor.
  auto* Y = Output(0);
  Y->CopyFrom(X);

  // Set output elements at even indices to zero.
  auto* Y_data = Y->mutable_data<float>();
  for (auto i = 0; i < Y->size(); i += 2) {
    Y_data[i] = 0.0f;
  }

  return true;
}

REGISTER_CPU_OPERATOR(ZeroEven, ZeroEvenOp<float, CPUContext>);

OPERATOR_SCHEMA(ZeroEven)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "1D input tensor")
    .Output(
        0,
        "Y",
        "1D output tensor");

} // namespace caffe2
