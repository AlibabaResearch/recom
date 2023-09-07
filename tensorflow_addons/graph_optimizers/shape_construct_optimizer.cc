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

#include <queue>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/types.pb.h>
#include <vector>

#include "shape_construct_optimizer.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

bool ShapeConstructOptimizer::OptimizeShapeInputs() {
  for (const auto &fc_node_set : fc_node_sets) {
    for (const std::string &node_name : fc_node_set) {
      NodeDef *node = node_mapping.at(node_name);

      std::vector<int> in_indices;
      bool has_shape_input = true;
      if (node->op() == "Reshape") {
        in_indices = {1};
      } else if (node->op() == "SparseReshape") {
        in_indices = {1, 2};
      } else if (node->op() == "SparseFillEmptyRows") {
        in_indices = {2};
      } else if (node->op() == "Tile") {
        in_indices = {1};
      } else {
        // TODO: add more cases
        has_shape_input = false;
      }

      if (has_shape_input) {
        for (int in_idx : in_indices) {
          NodeDef *inode = GetInputNode(node, in_idx);
          if (inode->op() != "Addons>ShapeConstruct") {
            RETURN_IF_FALSE(
                symbolic_context->ContentKnown(node->input(in_idx)));
            const ExprVec &shape_content =
                symbolic_context->GetContent(node->input(in_idx));
            NodeDef *shape_construct = ConstructShapeNodeByExpr(
                shape_content, GetInputType(node, in_idx),
                node_name + "_input_shape_" + std::to_string(in_idx));
            node->set_input(in_idx, shape_construct->name());
          }
        }
      }
    }
  }

  return true;
}

bool ShapeConstructOptimizer::PruneDeadShapeConstruct() {
  for (int i = 0; i < fc_node_sets.size(); ++i) {
    HashSetT<std::string> dead_nodes =
        GetDeadNodes(fc_node_sets[i], fc_boundary_node_sets[i]);
    RECOM_VLOG << "Get " << dead_nodes.size() << " dead nodes";
    if (RewriteShapeConstruct(fc_node_sets[i], dead_nodes)) {
      RECOM_VLOG << "RewriteShapeConstruct for FC " << i << " all successfully";
    } else {
      RECOM_VLOG << "RewriteShapeConstruct for FC " << i
                 << " are not all successful";
    }
  }

  return true;
}

HashSetT<std::string> ShapeConstructOptimizer::GetDeadNodes(
    const HashSetT<std::string> &fc_node_set,
    const HashSetT<std::string> &fc_boundary_node_set) {
  HashSetT<std::string> dead_nodes = fc_node_set;
  std::vector<std::string> node_stack;
  node_stack.insert(node_stack.begin(), fc_boundary_node_set.cbegin(),
                    fc_boundary_node_set.cend());
  while (!node_stack.empty()) {
    const std::string node_name = node_stack.back();
    node_stack.pop_back();
    dead_nodes.erase(node_name);
    if (symbolic_context->HasShapeNode(node_name)) {
      NodeDef *shape_node = symbolic_context->GetShapeNode(node_name);
      dead_nodes.erase(shape_node->name());
    }

    NodeDef *node = node_mapping.at(node_name);
    if (node->op() != "Addons>ShapeConstruct") {
      for (int i = 0; i < node->input_size(); ++i) {
        const std::string input_name = GetNodeNameByTensor(node->input(i));
        if (fc_node_set.count(input_name))
          node_stack.push_back(input_name);
      }
    }
  }

  return dead_nodes;
}

bool ShapeConstructOptimizer::RewriteShapeConstruct(
    const HashSetT<std::string> &fc_node_set,
    const HashSetT<std::string> &dead_nodes) {
  // TODO: update the ShapeConstruct
  bool all_success = true;
  for (const std::string &node_name : fc_node_set) {
    NodeDef *node = node_mapping.at(node_name);
    if (node->op() == "Addons>ShapeConstruct") {
      ExprVec dead_symbol_exprs;
      for (int i = 0; i < node->input_size(); ++i) {
        const std::string iname = GetNodeNameByTensor(node->input(i));
        if (dead_nodes.count(iname)) {
          const std::string symbol_expr_str =
              node->attr().at("symbols").list().s(i);
          dead_symbol_exprs.push_back(Expression(symbol_expr_str));
        }
      }
      if (dead_symbol_exprs.size() == 0)
        continue;

      RECOM_VLOG << "Found " << node->name() << " with dead symbols "
                 << ExprVecToStr(dead_symbol_exprs);

      ExprVec exprs(node->attr().at("exprs").list().s_size());
      for (int i = 0; i < exprs.size(); ++i) {
        exprs[i] = Expression(node->attr().at("exprs").list().s(i));
      }

      int success_cnt = 0;
      for (const Expression &dead_symbol_expr : dead_symbol_exprs) {
        std::vector<std::pair<Expression, NodeDef *>> eq_pairs =
            symbolic_context->FindEqSymbolExprGenNodePairs(dead_symbol_expr);
        bool success = false;
        for (auto eq_pair : eq_pairs) {
          NodeDef *gen_node = eq_pair.second;
          if (!dead_nodes.count(gen_node->name())) {
            RECOM_VLOG << "Replace " << dead_symbol_expr << " with "
                       << eq_pair.first;
            for (Expression &expr : exprs) {
              expr = expr.subs({{dead_symbol_expr, eq_pair.first}});
            }
            success = true;
            break;
          }
        }
        if (success)
          ++success_cnt;
      }

      if (success_cnt > 0) {
        RECOM_VLOG << "Reconstruct " << node->name();
        NodeDef *new_shape_construct = ConstructShapeNodeByExpr(
            exprs, node->attr().at("T").type(), node->name() + "_new");
        for (const std::string &oname : out_mapping.at(node->name())) {
          NodeDef *onode = node_mapping.at(oname);
          for (int i = 0; i < onode->input_size(); ++i) {
            if (GetNodeNameByTensor(onode->input(i)) == node->name()) {
              *(onode->mutable_input(i)) = new_shape_construct->name();
            }
          }
        }
      } else {
        RECOM_VLOG << "Fail to reconstruct " << node->name();
      }

      all_success = all_success && (success_cnt == dead_symbol_exprs.size());
    }
  }

  return all_success;
}

} // namespace feature_opt
} // namespace tensorflow