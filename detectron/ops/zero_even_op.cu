/**
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 */

#include "caffe2/core/context_gpu.h"

#include "zero_even_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SetEvenIndsToVal(size_t num_even_inds, T val, T* data) {
  CUDA_1D_KERNEL_LOOP(i, num_even_inds) {
    data[i << 1] = val;
  }
}

} // namespace

template <>
bool ZeroEvenOp<float, CUDAContext>::RunOnDevice() {
  // Retrieve the input tensor.
  const auto& X = Input(0);
  CAFFE_ENFORCE(X.ndim() == 1);

  // Initialize the output tensor to a copy of the input tensor.
  auto* Y = Output(0);
  Y->CopyFrom(X);

  // Set output elements at even indices to zero.
  auto output_size = Y->size();

  if (output_size > 0) {
    size_t num_even_inds = output_size / 2 + output_size % 2;
    SetEvenIndsToVal<float>
        <<<CAFFE_GET_BLOCKS(num_even_inds),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            num_even_inds,
            0.0f,
            Y->mutable_data<float>());
  }

  return true;
}

REGISTER_CUDA_OPERATOR(ZeroEven, ZeroEvenOp<float, CUDAContext>);

} // namespace caffe2
