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
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/tstring.h>

#include "gather_str_value_gen_indice_ops.h"

namespace tensorflow {
namespace feature_opt {

template <typename T>
GatherStrValueGenIndiceOp<T>::GatherStrValueGenIndiceOp(OpKernelConstruction *c)
    : OpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("not_equal", &not_equal));
  OP_REQUIRES_OK(c, c->GetAttr("dense_rank", &dense_rank));
}

template <typename T>
void GatherStrValueGenIndiceOp<T>::Compute(OpKernelContext *c) {
  const Tensor &values_t = c->input(0);
  const int nnz = values_t.NumElements();
  int num_output = 0;
  for (int i = 0; i < nnz; ++i) {
    if (values_t.flat<tstring>()(i) != not_equal)
      ++num_output;
  }

  Tensor *output_indices_t;
  OP_REQUIRES_OK(
      c, c->allocate_output(0, {num_output, dense_rank}, &output_indices_t));

  Tensor *output_values_t;
  OP_REQUIRES_OK(c, c->allocate_output(1, {num_output}, &output_values_t));

  for (int i = 0, j = 0; i < nnz; ++i) {
    if (values_t.flat<tstring>()(i) != not_equal) {
      output_indices_t->flat<T>()(j * dense_rank) = i;
      for (int k = 1; k < dense_rank; ++k) {
        output_indices_t->flat<T>()(j * dense_rank + k) = 0;
      }
      output_values_t->flat<tstring>()(j) = values_t.flat<tstring>()(i);
      ++j;
    }
  }
}

REGISTER_KERNEL_BUILDER(Name("Addons>GatherStrValueGenIndice")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32>("T"),
                        GatherStrValueGenIndiceOp<int>);

REGISTER_KERNEL_BUILDER(Name("Addons>GatherStrValueGenIndice")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int64>("T"),
                        GatherStrValueGenIndiceOp<int64>);

REGISTER_OP("Addons>GatherStrValueGenIndice")
    .Input("values: string")
    .Output("output_indices: T")
    .Output("output_values: string")
    .Attr("dense_rank: int = 1")
    .Attr("not_equal: string = ''")
    .Attr("T: {int32, int64} = DT_INT64")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int dense_rank;
      TF_RETURN_IF_ERROR(c->GetAttr("dense_rank", &dense_rank));
      c->set_output(0, c->Matrix(c->kUnknownDim, dense_rank));
      c->set_output(1, c->Vector(c->kUnknownDim));

      return Status::OK();
    });

} // namespace feature_opt
} // namespace tensorflow