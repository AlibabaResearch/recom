// Copyright 2023 The RECom Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <symengine/expression.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_requires.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/errors.h>

#include "gather_indice_value_ops.h"

namespace tensorflow {
namespace feature_opt {

template <typename T>
GatherIndiceValueOp<T>::GatherIndiceValueOp(OpKernelConstruction *c)
    : OpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("left_boundaries", &left_boundaries));
  OP_REQUIRES_OK(c, c->GetAttr("right_boundaries", &right_boundaries));
  OP_REQUIRES(c, left_boundaries.size() == right_boundaries.size(),
              errors::InvalidArgument(
                  "left_boundaries.size() != right_boundaries.size()"));
}

template <typename T> void GatherIndiceValueOp<T>::Compute(OpKernelContext *c) {
  const int n = left_boundaries.size();
  auto comp = [&](T x) -> bool {
    for (int i = 0; i < n; ++i) {
      if (x >= left_boundaries[i] || x <= right_boundaries[i])
        return true;
    }
    return false;
  };

  const Tensor &indices_t = c->input(0);
  const Tensor &values_t = c->input(1);
  const int nnz = values_t.NumElements();
  int num_output = 0;
  for (int i = 0; i < nnz; ++i) {
    if (comp(values_t.flat<T>()(i)))
      ++num_output;
  }

  const int rank = indices_t.shape().dim_size(1);
  Tensor *output_indices_t;
  OP_REQUIRES_OK(c,
                 c->allocate_output(0, {num_output, rank}, &output_indices_t));

  Tensor *output_values_t;
  OP_REQUIRES_OK(c, c->allocate_output(1, {num_output}, &output_values_t));

  for (int i = 0, j = 0; i < nnz; ++i) {
    if (comp(values_t.flat<T>()(i))) {
      for (int k = 0; k < rank; ++k) {
        output_indices_t->flat<T>()(j * rank + k) =
            indices_t.flat<T>()(i * rank + k);
      }
      output_values_t->flat<T>()(j) = values_t.flat<T>()(i);
      ++j;
    }
  }
}

REGISTER_KERNEL_BUILDER(Name("Addons>GatherIndiceValue")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32>("T"),
                        GatherIndiceValueOp<int>);

REGISTER_KERNEL_BUILDER(Name("Addons>GatherIndiceValue")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int64>("T"),
                        GatherIndiceValueOp<int64>);

REGISTER_OP("Addons>GatherIndiceValue")
    .Input("indices: T")
    .Input("values: T")
    .Output("output_indices: T")
    .Output("output_values: T")
    .Attr("left_boundaries: list(int)")
    .Attr("right_boundaries: list(int)")
    .Attr("T: {int32, int64} = DT_INT64");

} // namespace feature_opt
} // namespace tensorflow