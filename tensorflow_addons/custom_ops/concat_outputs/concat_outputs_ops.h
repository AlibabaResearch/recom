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

#pragma once
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace feature_opt {

template <typename T> class ConcatOutputsOp : public OpKernel {
private:
  int N;
  int BLOCK_THREADS;
  int prefix_begin, prefix_end;
  std::vector<int> embedd_dims;
  std::vector<int> device_input_indices;
  std::vector<int> device_concat_indices;
  std::vector<int> host_concat_indices;
  std::string dlpath;

  int embedd_dim_sum;

  void (*ConcatOutputs)(T **, T **, T *, int, cudaStream_t);

public:
  explicit ConcatOutputsOp(OpKernelConstruction *c);

  void Compute(OpKernelContext *c) override;
};

} // namespace feature_opt
} // namespace tensorflow