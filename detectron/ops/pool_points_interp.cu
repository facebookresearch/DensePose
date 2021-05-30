/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "pool_points_interp.h"

#include <stdio.h>

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__
float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width, T x, T y,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (x < -1.0 || x > width || y < -1.0 || y > height) {
    return 0;
  }

  if (x <= 0) x = 0;
  if (y <= 0) y = 0;

  int x_low = (int) x;
  int y_low = (int) y;
  int x_high, y_high;

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  // lambdas in X, Y axes
  T lx = x - x_low, ly = y - y_low;
  T hx = 1. - lx, hy = 1. - ly;

  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];   // top-left point
  T v2 = bottom_data[y_low * width + x_high];  // top-right point
  T v3 = bottom_data[y_high * width + x_low];  // bottom-left point
  T v4 = bottom_data[y_high * width + x_high]; // bottom-right point
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
__global__ void PoolPointsInterpForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels, const int height, const int width,
    const T* coords, T* top_data) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c) is an element in the pooled/interpolated output
    int c = index % channels;
    int n = index / channels;

    const T* offset_coords = coords + n * 2;
    // Get index of current fg roi among all fg rois in a minibatch
    int roi_batch_ind = n / 196;
    // Get spatial coordinate (x, y)
    T x = offset_coords[0] * spatial_scale;
    T y = offset_coords[1] * spatial_scale;
    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
    // Compute interpolated value
    top_data[index] = bilinear_interpolate(
        offset_bottom_data, height, width, x, y, index);
  }
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, T x, T y,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (x < -1.0 || x > width || y < -1.0 || y > height) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (x <= 0) x = 0;
  if (y <= 0) y = 0;

  x_low = (int) x;
  y_low = (int) y;

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  T lx = x - x_low, ly = y - y_low;
  T hx = 1. - lx, hy = 1. - ly;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void PoolPointsInterpBackward(const int nthreads, const T* top_diff,
    const int num_rois, const T spatial_scale, const int channels,
    const int height, const int width, T* bottom_diff, const T* coords) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int n = index / channels;

    const T* offset_coords = coords + n * 2;
    int roi_batch_ind = n / 196;
    T x = offset_coords[0] * spatial_scale;
    T y = offset_coords[1] * spatial_scale;

    T w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;

    bilinear_interpolate_gradient(height, width, x, y,
        w1, w2, w3, w4, x_low, x_high, y_low, y_high, index);

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    //
    int top_offset = (n * channels + c) ;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[0];
    //
    T g1 = top_diff_this_bin * w1 ;
    T g2 = top_diff_this_bin * w2 ;
    T g3 = top_diff_this_bin * w3 ;
    T g4 = top_diff_this_bin * w4 ;
    //
    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
    {
      gpu_atomic_add(static_cast<T>(g1), offset_bottom_diff + y_low * width + x_low);
      gpu_atomic_add(static_cast<T>(g2), offset_bottom_diff + y_low * width + x_high);
      gpu_atomic_add(static_cast<T>(g3), offset_bottom_diff + y_high * width + x_low);
      gpu_atomic_add(static_cast<T>(g4), offset_bottom_diff + y_high * width + x_high);
    } // if

  } // CUDA_1D_KERNEL_LOOP
} // PoolPointsInterpBackward

} // namespace

template<>
bool PoolPointsInterpOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data to pool
  auto& R = Input(1);  // Spatial coordinates of all points within RoIs

  if (R.size() == 0) {
    // Handle empty rois
    std::vector<int64_t> sizes = {0, X.dim32(1)};
    /* auto* Y = */ Output(0, sizes, at::dtype<float>());
    return true;
  }

  auto* Y = Output(0, {R.dim32(0), X.dim32(1)}, at::dtype<float>());  // Pooled interpolated data
  int output_size = Y->size();
  PoolPointsInterpForward<float><<<CAFFE_GET_BLOCKS(output_size),
                                  CAFFE_CUDA_NUM_THREADS,
                                  0, context_.cuda_stream()>>>(
      output_size, X.data<float>(), spatial_scale_,
      X.dim32(1), X.dim32(2), X.dim32(3),
      R.data<float>(), Y->mutable_data<float>());
  return true;
}

namespace {    
template <typename T>
__global__ void SetKernel(const int N, const T alpha, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = alpha;
  }
}
} // namespace

namespace {    
template <typename T>
__global__ void SetEvenIndsToVal(size_t num_even_inds, T val, T* data) {
  CUDA_1D_KERNEL_LOOP(i, num_even_inds) {
    data[i << 1] = val;
  }
}
} // namespace

template<>
bool PoolPointsInterpGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X  = Input(0);  // Input data to pool
  auto& R  = Input(1);  // 2D Spatial coordinates of all points within RoIs
  auto& dY = Input(2);  // Gradient of net w.r.t. output of "forward" op
                        // (aka "gradOutput")
  auto* dX = Output(
    0, X.sizes(), at::dtype<float>());  // Gradient of net w.r.t. input to
                                        // "forward" op (aka "gradInput")

  SetKernel<float><<<CAFFE_GET_BLOCKS(dX->size()),
                    CAFFE_CUDA_NUM_THREADS,
                    0, context_.cuda_stream()>>>(
      dX->size(), 0.f, dX->mutable_data<float>());

  if (dY.size() > 0) {  // Handle possibly empty gradient if there were no rois
    PoolPointsInterpBackward<float><<<CAFFE_GET_BLOCKS(dY.size()),
                                      CAFFE_CUDA_NUM_THREADS,
                                      0, context_.cuda_stream()>>>(
        dY.size(), dY.data<float>(), R.dim32(0), spatial_scale_,
        X.dim32(1), X.dim32(2), X.dim32(3),
        dX->mutable_data<float>(), R.data<float>());
  }
  return true;
}


REGISTER_CUDA_OPERATOR(PoolPointsInterp,
                       PoolPointsInterpOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PoolPointsInterpGradient,
                       PoolPointsInterpGradientOp<float, CUDAContext>);
} // namespace caffe2
