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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stack>
#include <vector>

#include "fc_optimizer_base.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

class ShapeConstructOptimizer : public FCOptimizerBase {
public:
  ShapeConstructOptimizer(GraphInfo &ginfo) : FCOptimizerBase(ginfo) {}

  bool OptimizeShapeInputs();

  bool PruneDeadShapeConstruct();

private:
  bool Optimize(const HashSetT<std::string> &fc_node_set,
                const HashSetT<std::string> &fc_boundary_node_set);

  bool OptimizeReshape(NodeDef *node);

  bool IsReshape(NodeDef *node, ExprVec &new_shape, int &in_idx,
                 DataType &type);

  HashSetT<std::string>
  GetDeadNodes(const HashSetT<std::string> &fc_node_set,
               const HashSetT<std::string> &fc_boundary_node_set);

  bool RewriteShapeConstruct(const HashSetT<std::string> &fc_node_set,
                             const HashSetT<std::string> &dead_nodes);
};

} // namespace feature_opt
} // namespace tensorflow