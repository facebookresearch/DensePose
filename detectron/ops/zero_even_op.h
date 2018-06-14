/**
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
 */

#ifndef ZERO_EVEN_OP_H_
#define ZERO_EVEN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
//#include "caffe2/utils/math.h"

namespace caffe2 {

/**
 * ZeroEven operator. Zeros elements at even indices of an 1D array.
 * Elements at odd indices are preserved.
 *
 * This toy operator is an example of a custom operator and may be a useful
 * reference for adding new custom operators to the Detectron codebase.
 */
template <typename T, class Context>
class ZeroEvenOp final : public Operator<Context> {
 public:
  // Introduce Operator<Context> helper members.
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ZeroEvenOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // ZERO_EVEN_OP_H_
