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

namespace functor {

template <typename Device, typename T, typename Tindices, typename Tshape>
class ExtendedSparseToDenseFunctor {
public:
  void operator()(OpKernelContext *c, const Tensor &indices_t,
                  const Tensor &values_t, const Tensor &dense_prefix_t,
                  T default_value);
};

} // namespace functor

} // namespace feature_opt
} // namespace tensorflow