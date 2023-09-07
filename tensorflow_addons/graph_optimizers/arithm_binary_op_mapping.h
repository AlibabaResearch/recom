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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

template <typename Dtype> class ArithmBinaryOpMappingTemplate {
  HashMapT<std::string, std::function<Dtype(Dtype, Dtype)>> arithm_mapping;

public:
  ArithmBinaryOpMappingTemplate() {
    arithm_mapping["Add"] = [](Dtype a, Dtype b) -> Dtype { return a + b; };
    arithm_mapping["AddV2"] = [](Dtype a, Dtype b) -> Dtype { return a + b; };
    arithm_mapping["Sub"] = [](Dtype a, Dtype b) -> Dtype { return a - b; };
    arithm_mapping["Mul"] = [](Dtype a, Dtype b) -> Dtype { return a * b; };
    arithm_mapping["Div"] = [](Dtype a, Dtype b) -> Dtype { return a / b; };
  }

  bool count(const std::string &op) { return arithm_mapping.count(op); }

  std::function<Dtype(Dtype, Dtype)> operator[](const std::string &op) {
    return arithm_mapping[op];
  }
};

class ArithmBinaryOpMapping {
  ArithmBinaryOpMappingTemplate<int> int_arithm_mapping;
  ArithmBinaryOpMappingTemplate<int32> int32_arithm_mapping;
  ArithmBinaryOpMappingTemplate<int64> int64_arithm_mapping;
  ArithmBinaryOpMappingTemplate<float> float_arithm_mapping;
  ArithmBinaryOpMappingTemplate<double> double_arithm_mapping;

public:
  ArithmBinaryOpMapping() {}

  template <typename Dtype = int> bool count(const std::string &op) {
    return int_arithm_mapping.count(op);
  }

  template <typename Dtype>
  std::function<Dtype(Dtype, Dtype)> map(const std::string &op) {
    if (std::is_same<Dtype, int>::value)
      return int_arithm_mapping[op];
    if (std::is_same<Dtype, int32>::value)
      return int32_arithm_mapping[op];
    if (std::is_same<Dtype, int64>::value)
      return int64_arithm_mapping[op];
    if (std::is_same<Dtype, float>::value)
      return float_arithm_mapping[op];
    if (std::is_same<Dtype, double>::value)
      return double_arithm_mapping[op];
    exit(-1);
  }
};

} // namespace feature_opt
} // namespace tensorflow