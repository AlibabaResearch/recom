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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace feature_opt {

template <typename T> class ShapeConstructOp : public OpKernel {
private:
  std::vector<DataType> input_types;
  std::vector<std::string> symbols;
  std::vector<int> indices;
  std::vector<std::string> expr_strings;
  std::string dlpath;

  void (*ShapeConstruct)(const std::vector<int64> &, void *);

public:
  explicit ShapeConstructOp(OpKernelConstruction *c);

  void Compute(OpKernelContext *c) override;
};

} // namespace feature_opt
} // namespace tensorflow