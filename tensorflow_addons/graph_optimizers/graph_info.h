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
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn_registry.h"

namespace tensorflow {
namespace feature_opt {

using grappler::GraphProperties;

struct GraphInfo {
  grappler::GrapplerItem &item;
  GraphDef *gd;

  OutMap out_mapping;
  NodeMap node_mapping;
  HashMapT<std::string, int> indegree_mapping;
  HashMapT<std::string, int> nonconst_indegree_mapping;
  std::vector<HashSetT<std::string>> fc_node_sets;
  std::vector<HashSetT<std::string>> fc_boundary_node_sets;

  std::unique_ptr<GraphProperties> properties;
  std::shared_ptr<SymbolicShapeContext> symbolic_context;

  GraphInfo(grappler::GrapplerItem &item);

  bool IsConcatOutOp(NodeDef *node) const;

  bool UpdateAll();

  bool UpdateProperties();

  bool ReadInitConfig();

  bool InitSymbolicShape();

  bool SymbolicShapePropagation();

  bool UpdateTopoInfo();

  bool ExtractFCNodes();

  bool PruneFCDeadNodes();

  bool RenameFCNodes();

  GraphInfo(const GraphInfo &) = delete;
  GraphInfo(GraphInfo &&) = delete;
};

} // namespace feature_opt
} // namespace tensorflow