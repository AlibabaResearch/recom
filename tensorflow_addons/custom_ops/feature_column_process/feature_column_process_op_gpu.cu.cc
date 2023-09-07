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

#define EIGEN_USE_GPU

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <numeric>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_requires.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/platform/errors.h>
#include <vector>

#include "feature_column_process_ops.h"

namespace tensorflow {
namespace feature_opt {

FeatureColumnProcessOp::FeatureColumnProcessOp(OpKernelConstruction *c)
    : OpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("input_types", &input_types));
  OP_REQUIRES_OK(c, c->GetAttr("output_types", &output_types));
  OP_REQUIRES_OK(c, c->GetAttr("input_ranks", &input_ranks));
  OP_REQUIRES_OK(c, c->GetAttr("output_ranks", &output_ranks));
  OP_REQUIRES(
      c, input_ranks.size() == input_types.size(),
      errors::InvalidArgument("input_ranks.size() != input_types.size()"));
  OP_REQUIRES(
      c, output_ranks.size() == output_types.size(),
      errors::InvalidArgument("output_ranks.size() != output_types.size()"));

  input_rank_sum = std::accumulate(input_ranks.begin(), input_ranks.end(), 0,
                                   std::plus<int>());
  output_rank_sum = std::accumulate(output_ranks.begin(), output_ranks.end(), 0,
                                    std::plus<int>());

  std::string dlpath;
  OP_REQUIRES_OK(c, c->GetAttr("dlpath", &dlpath));

  void *handle = dlopen(dlpath.c_str(), RTLD_NOW);
  OP_REQUIRES(c, handle, errors::Aborted("Fail to dlopen " + dlpath));

  auto CreateConstBuffers =
      reinterpret_cast<void (*)(char *&)>(dlsym(handle, "CreateConstBuffers"));
  CreateConstBuffers(const_buff);

  ProcessFeatureColumns = reinterpret_cast<decltype(ProcessFeatureColumns)>(
      dlsym(handle, "ProcessFeatureColumns"));
}

void FeatureColumnProcessOp::Compute(OpKernelContext *c) {
  cudaStream_t strm = c->eigen_gpu_device().stream();

  std::vector<void *> input_ptrs(input_ranks.size());
  std::vector<void *> output_ptrs(output_ranks.size());
  std::vector<int> input_shapes(input_rank_sum);
  std::vector<int> output_shapes(output_rank_sum);

  Tensor *output_ptrs_tensor;
  OP_REQUIRES_OK(c,
                 c->allocate_output(0, {static_cast<int>(output_ptrs.size())},
                                    &output_ptrs_tensor));

  Tensor *output_shapes_tensor;
  OP_REQUIRES_OK(
      c, c->allocate_output(1, {output_rank_sum}, &output_shapes_tensor));

  auto input_shape_itr = input_shapes.begin();
  for (int i = 0; i < input_ptrs.size(); ++i) {
    const Tensor &input_t = c->input(3 + i);
    input_ptrs[i] = input_t.data();

    const TensorShape &input_shape = input_t.shape();
    for (int j = 0; j < input_shape.dims(); ++j) {
      *input_shape_itr = input_shape.dim_size(j);
      ++input_shape_itr;
    }
  }

  const int *symbol_input = nullptr;
  if (c->num_inputs() == input_types.size() + 4) {
    symbol_input = c->input(input_types.size() + 3).flat<int>().data();
  }

  std::vector<Tensor *> temp_tensors;
  auto malloc_temp = [&](void **p, int size) -> void {
    Tensor *t = new Tensor;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT8, {size}, t));
    *p = t->data();
    temp_tensors.push_back(t);
  };

  auto malloc_buff = [&](void **p, int size) -> void {
    Tensor *t;
    OP_REQUIRES_OK(c, c->allocate_output(2, {size}, &t));
    *p = t->data();
  };

  ProcessFeatureColumns(const_buff, c->input(0).flat<int8>().data(),
                        c->input(1).flat<int>().data(),
                        c->input(2).flat<int>().data(), input_ptrs,
                        input_shapes, symbol_input, output_ptrs, output_shapes,
                        strm, malloc_temp, malloc_buff);

  CubDebugExit(cudaMemcpyAsync(output_ptrs_tensor->data(), &output_ptrs[0],
                               output_ptrs.size() * sizeof(void *),
                               cudaMemcpyHostToDevice, strm));

  CubDebugExit(cudaStreamSynchronize(strm));

  memcpy(output_shapes_tensor->data(), &output_shapes[0],
         output_shapes.size() * sizeof(int));

  for (auto t : temp_tensors) {
    delete t;
  }
}

REGISTER_KERNEL_BUILDER(Name("Addons>FeatureColumnProcess")
                            .Device(tensorflow::DEVICE_GPU)
                            .HostMemory("concated_offsets")
                            .HostMemory("concated_shapes")
                            .HostMemory("output_shapes"),
                        FeatureColumnProcessOp);

REGISTER_OP("Addons>FeatureColumnProcess")
    .Input("concated_inputs: int8")
    .Input("concated_offsets: int32")
    .Input("concated_shapes: int32")
    .Input("inputs: input_types")
    .Output("output_ptrs: int64")
    .Output("output_shapes: int32")
    .Output("buffer: int8")
    .Attr("input_types: list(type)")
    .Attr("output_types: list(type)")
    .Attr("input_ranks: list(int)")
    .Attr("output_ranks: list(int)")
    .Attr("dlpath: string");

REGISTER_KERNEL_BUILDER(Name("Addons>FeatureColumnProcessWithSymbols")
                            .Device(tensorflow::DEVICE_GPU)
                            .HostMemory("concated_offsets")
                            .HostMemory("concated_shapes")
                            .HostMemory("symbols")
                            .HostMemory("output_shapes"),
                        FeatureColumnProcessOp);

REGISTER_OP("Addons>FeatureColumnProcessWithSymbols")
    .Input("concated_inputs: int8")
    .Input("concated_offsets: int32")
    .Input("concated_shapes: int32")
    .Input("inputs: input_types")
    .Input("symbols: int32")
    .Output("output_ptrs: int64")
    .Output("output_shapes: int32")
    .Output("buffer: int8")
    .Attr("input_types: list(type)")
    .Attr("output_types: list(type)")
    .Attr("input_ranks: list(int)")
    .Attr("output_ranks: list(int)")
    .Attr("dlpath: string");

} // namespace feature_opt
} // namespace tensorflow