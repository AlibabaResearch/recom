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

#include <cstring>
#include <functional>
#include <numeric>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_requires.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/platform/errors.h>
#include <vector>

#include "concat_inputs_ops.h"

namespace tensorflow {
namespace feature_opt {

ConcatInputsOp::ConcatInputsOp(OpKernelConstruction *c) : OpKernel(c) {
  std::vector<DataType> input_types;
  OP_REQUIRES_OK(c, c->GetAttr("T", &input_types));

  std::vector<int> input_ranks;
  OP_REQUIRES_OK(c, c->GetAttr("ranks", &input_ranks));
  OP_REQUIRES(
      c, input_types.size() == input_ranks.size(),
      errors::InvalidArgument("input_types.size() != input_ranks.size()"));

  num_inputs = input_types.size();
  num_shape_elements = std::accumulate(input_ranks.begin(), input_ranks.end(),
                                       0, std::plus<int>());
}

void ConcatInputsOp::Compute(OpKernelContext *c) {
  Tensor *offsets_tensor;
  OP_REQUIRES_OK(c, c->allocate_output(1, {num_inputs}, &offsets_tensor));

  Tensor *shapes_tensor;
  OP_REQUIRES_OK(c,
                 c->allocate_output(2, {num_shape_elements}, &shapes_tensor));

  std::vector<int> input_mem_sizes(num_inputs);

  int output_size = 0;
  int *offsets_data = offsets_tensor->flat<int>().data();
  int *shapes_data_itr = shapes_tensor->flat<int>().data();
  for (int i = 0; i < num_inputs; ++i) {
    const Tensor &input_tensor = c->input(i);
    offsets_data[i] = output_size;
    input_mem_sizes[i] =
        input_tensor.NumElements() * DataTypeSize(input_tensor.dtype());
    output_size += input_mem_sizes[i];

    const TensorShape &input_shape = input_tensor.shape();
    for (int j = 0; j < input_shape.dims(); ++j) {
      *(shapes_data_itr++) = input_shape.dim_size(j);
    }
  }

  Tensor *output_tensor;
  OP_REQUIRES_OK(c, c->allocate_output(0, {output_size}, &output_tensor));

  int8 *output_data_itr = output_tensor->flat<int8>().data();
  for (int i = 0; i < num_inputs; ++i) {
    const Tensor &input_tensor = c->input(i);
    mempcpy(output_data_itr, input_tensor.data(), input_mem_sizes[i]);
    output_data_itr += input_mem_sizes[i];
  }
}

REGISTER_KERNEL_BUILDER(
    Name("Addons>ConcatInputs").Device(tensorflow::DEVICE_CPU), ConcatInputsOp);

REGISTER_OP("Addons>ConcatInputs")
    .Input("inputs: T")
    .Output("output: int8")
    .Output("offsets: int32")
    .Output("shapes: int32")
    .Attr("T: list(type)")
    .Attr("ranks: list(int)");

} // namespace feature_opt
} // namespace tensorflow