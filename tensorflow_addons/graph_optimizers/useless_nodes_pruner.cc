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

#include <functional>
#include <numeric>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/types.pb.h>
#include <vector>

#include "useless_nodes_pruner.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

void UselessNodesPruner::Optimize() {
  for (int i = 0; i < fc_node_sets.size(); ++i) {
    for (const std::string &node_name : fc_node_sets[i]) {
      NodeDef *node = node_mapping.at(node_name);
      int useful_idx;
      if (MatchIdentity(node, useful_idx) ||
          MatchUselessTranspose(node, useful_idx) ||
          MatchUselessArithm(node, useful_idx) ||
          MatchUselessStridedSlice(node, useful_idx)) {
        for (const std::string &output_name : out_mapping.at(node_name)) {
          NodeDef *output_node = node_mapping.at(output_name);
          for (int i = 0; i < output_node->input_size(); ++i) {
            if (GetNodeNameByTensor(output_node->input(i)) == node_name) {
              *(output_node->mutable_input(i)) = node->input(useful_idx);
            }
          }
        }
        out_mapping[GetNodeNameByTensor(node->input(useful_idx))].insert(
            out_mapping.at(node_name).begin(), out_mapping.at(node_name).end());
        RECOM_VLOG << "Prune " << node_name;
      }
    }
  }
}

bool UselessNodesPruner::MatchIdentity(NodeDef *node, int &useful_idx) {
  if (node->op() == "Identity") {
    for (int i = 0; i < node->input_size(); ++i) {
      if (node->input(0)[0] == '^')
        return false;
    }
    useful_idx = 0;
    return true;
  }

  return false;
}

bool UselessNodesPruner::MatchUselessTranspose(NodeDef *node, int &useful_idx) {
  if (node->op() == "Transpose") {
    std::vector<int> perm;
    if (symbolic_context->ContentStatic(node->input(1), perm)) {
      for (int i = 0; i < perm.size(); ++i) {
        if (perm[i] != i) {
          return false;
        }
      }
      useful_idx = 0;
      return true;
    }
  }

  return false;
}

bool UselessNodesPruner::MatchUselessArithm(NodeDef *node, int &useful_idx) {
  if (arithm_mapping.count(node->op())) {
    int useful_idx_tmp = -1;
    for (int i = 0; i < 2 && useful_idx_tmp < 0; ++i) {
      double c;
      if (ExtractSplatConst(GetInputNode(node, i), c)) {
        if (c == 0) {
          if (node->op() == "Add" || node->op() == "AddV2") {
            useful_idx_tmp = 1 - i;
          } else if (node->op() == "Sub") {
            if (i == 1) {
              useful_idx_tmp = 0;
            }
          }
        } else if (c == 1) {
          if (node->op() == "Mul") {
            useful_idx_tmp = 1 - i;
          } else if (node->op() == "Div") {
            if (i == 1) {
              useful_idx_tmp = 0;
            }
          }
        }
      }
    }

    if (useful_idx_tmp >= 0) {
      // avoid broadcast
      RETURN_IF_FALSE(symbolic_context->ShapeKnown(node->name()));
      RETURN_IF_FALSE(
          symbolic_context->ShapeKnown(node->input(useful_idx_tmp)));
      RETURN_IF_FALSE(symbolic_context->IsEq(
          symbolic_context->GetShape(node->name()),
          symbolic_context->GetShape(node->input(useful_idx_tmp))));
      useful_idx = useful_idx_tmp;
      return true;
    }
  }

  return false;
}

bool UselessNodesPruner::MatchUselessStridedSlice(NodeDef *node,
                                                  int &useful_idx) {
  if (node->op() == "StridedSlice") {
    if (symbolic_context->ShapeKnown(node->input(0)) &&
        symbolic_context->ShapeKnown(node->name())) {
      const ExprVec input_shape = symbolic_context->GetShape(node->input(0));
      const ExprVec output_shape = symbolic_context->GetShape(node->name());
      if (symbolic_context->IsEq(input_shape, output_shape)) {
        useful_idx = 0;
        return true;
      }
    }
  }

  return false;
}

} // namespace feature_opt
} // namespace tensorflow