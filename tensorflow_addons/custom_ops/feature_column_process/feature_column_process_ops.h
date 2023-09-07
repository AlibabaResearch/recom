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
#include <cub/cub.cuh>
#include <functional>
#include <string>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/types.pb.h>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace feature_opt {

class FeatureColumnProcessOp : public OpKernel {
private:
  std::vector<DataType> input_types;
  std::vector<DataType> output_types;
  std::vector<int> input_ranks;
  std::vector<int> output_ranks;
  int input_rank_sum;
  int output_rank_sum;

  char *const_buff;

  void (*ProcessFeatureColumns)(const char *, const int8 *, const int *,
                                const int *, const std::vector<void *> &,
                                const std::vector<int> &, const int *,
                                std::vector<void *> &, std::vector<int> &,
                                const cudaStream_t &,
                                const std::function<void(void **, int)> &,
                                const std::function<void(void **, int)> &);

public:
  explicit FeatureColumnProcessOp(OpKernelConstruction *c);
  ~FeatureColumnProcessOp() { CubDebugExit(cudaFree(const_buff)); }

  void Compute(OpKernelContext *c) override;
};

} // namespace feature_opt
} // namespace tensorflow