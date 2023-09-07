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
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stack>
#include <unordered_map>

#include "fc_optimizer_base.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

class PostLookupOptimizer : public FCOptimizerBase {
public:
  PostLookupOptimizer(GraphInfo &ginfo) : FCOptimizerBase(ginfo) {}

  void Optimize();

private:
  struct ExSpInfo {
    ExprVec dense_prefix;
    ExprVec value_shape;
    ExprVec dense_shape;
    Expression element_size;
    float default_value;
  };

  bool GetExpectDensePrefixAndValueShape(const ExSpInfo &info,
                                         ExprVec &expect_prefix,
                                         ExprVec &expect_value_shape);

  struct PostGraphContext {
    HashMapT<std::string, int> node_cnt_mapping;
    HashMapT<std::string, int> curr_cnt_mapping;
    HashMapT<std::string, std::map<int, std::pair<NodeDef *, ExSpInfo>>>
        pending_nodes;
    int to_dense_cpy_cnt;
  };

  void InitPostGraphContext(NodeDef *to_dense_node, int i,
                            PostGraphContext &context);

  bool Optimize(NodeDef *to_dense_node, NodeDef *onode, int onode_inid,
                ExSpInfo curr, PostGraphContext &context);

  bool MatchAndRecordReshape(NodeDef *to_dense_node, NodeDef *onode,
                             int onode_inid, ExSpInfo &curr,
                             PostGraphContext &context);

  bool MatchAndRewriteMatMul(NodeDef *to_dense_node, NodeDef *onode,
                             int onode_inid, ExSpInfo &curr,
                             PostGraphContext &context);

  bool MatchAndRecordSelect(NodeDef *to_dense_node, NodeDef *onode,
                            int onode_inid, ExSpInfo &curr,
                            PostGraphContext &context);

  bool MatchAndRewriteSoftmax(NodeDef *to_dense_node, NodeDef *onode,
                              int onode_inid, ExSpInfo &curr,
                              PostGraphContext &context);

  bool MatchAndRewriteMul(NodeDef *to_dense_node, NodeDef *onode,
                          int onode_inid, ExSpInfo &curr,
                          PostGraphContext &context);

  bool MatchShapeAndReconstruct(NodeDef *to_dense_node, NodeDef *onode,
                                int onode_inid, ExSpInfo &curr,
                                PostGraphContext &context);

  bool ReconstructToDense(NodeDef *to_dense_node, NodeDef *onode,
                          int onode_inid, ExSpInfo &curr,
                          PostGraphContext &context);
};

} // namespace feature_opt
} // namespace tensorflow