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
#include <memory>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

#include "symbolic_shape_fn.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

class SymbolicShapeFnRegistry {
  HashMapT<std::string, std::unique_ptr<SymbolicShapeFn>> op_fn_mapping;

public:
  static SymbolicShapeFnRegistry *Global();

  void Register(const std::string op, std::unique_ptr<SymbolicShapeFn> fn);

  bool OpRegistered(const std::string &op) {
    return op_fn_mapping.count(op);
  }

  bool Run(std::shared_ptr<SymbolicShapeContext> context, NodeDef *node);
};

namespace symbolic_fn_registration {

class SymbolicShapeFnRegistration {
public:
  SymbolicShapeFnRegistration(const std::string op,
                              std::unique_ptr<SymbolicShapeFn> fn) {
    SymbolicShapeFnRegistry::Global()->Register(op, std::move(fn));
  }
};

} // namespace symbolic_fn_registration

bool RunSymbolicFn(std::shared_ptr<SymbolicShapeContext> context,
                   NodeDef *node);

#define REGISTER_SYMBOLIC_SHAPE_FN(op, ...)                                    \
  REGISTER_SYMBOLIC_SHAPE_FN_UNIQ_HELPER(__COUNTER__, op, __VA_ARGS__)

#define REGISTER_SYMBOLIC_SHAPE_FN_UNIQ_HELPER(ctr, op, ...)                   \
  REGISTER_SYMBOLIC_SHAPE_FN_UNIQ(ctr, op, __VA_ARGS__)

#define REGISTER_SYMBOLIC_SHAPE_FN_UNIQ(ctr, op, ...)                          \
  static ::tensorflow::feature_opt::symbolic_fn_registration::                 \
      SymbolicShapeFnRegistration register_symbolic_shape_##ctr(               \
          op, ::std::unique_ptr<::tensorflow::feature_opt::SymbolicShapeFn>(   \
                  new __VA_ARGS__()))

} // namespace feature_opt
} // namespace tensorflow