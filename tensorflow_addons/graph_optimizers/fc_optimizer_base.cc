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

#include <cstdio>
#include <fstream>
#include <stack>
#include <string>
#include <symengine/expression.h>

#include "fc_optimizer_base.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn_registry.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

NodeDef *FCOptimizerBase::FindOutputNode(const NodeDef *node,
                                         const std::string &op_types) {
  std::vector<std::string> tvec = absl::StrSplit(op_types, ",");
  if (!out_mapping.count(node->name())) {
    return nullptr;
  }

  for (const std::string &oname : out_mapping.at(node->name())) {
    NodeDef *onode = node_mapping.at(oname);
    if (find(tvec.begin(), tvec.end(), onode->op()) != tvec.end()) {
      return onode;
    }
  }
  return nullptr;
}

NodeDef *FCOptimizerBase::FindInputNodeByIdx(const NodeDef *node, const int idx,
                                             const std::string &op_types) {
  std::vector<std::string> tvec = absl::StrSplit(op_types, ",");
  NodeDef *inode = node_mapping.at(GetNodeNameByTensor(node->input(idx)));
  if (find(tvec.begin(), tvec.end(), inode->op()) != tvec.end() ||
      op_types.empty()) {
    return inode;
  }
  RECOM_VLOG_WARNING << "Cannot find {" << op_types << "} at " << node->name()
                     << " " << idx << ". Found " << inode->op();
  return nullptr;
}

DataType FCOptimizerBase::GetInputType(const NodeDef *node,
                                       const int input_idx) {
  const auto &input = properties.GetInputProperties(node->name());
  int idx = input_idx;
  if (idx >= input.size()) {
    RECOM_VLOG_WARNING << "Clipping input idx " << idx << " of node "
                       << node->name() << " to " << input.size() - 1;
    idx = input_idx - 1;
  }
  if (idx < 0) {
    RECOM_VLOG_WARNING << "Clipping input idx " << idx << " of node "
                       << node->name() << " to " << 0;
    idx = 0;
  }
  const DataType type = input[idx].dtype();
  return type;
}

DataType FCOptimizerBase::GetOutputType(const NodeDef *node,
                                        const int output_idx) {
  const auto &output = properties.GetOutputProperties(node->name());
  int idx = output_idx;
  if (idx >= output.size()) {
    RECOM_VLOG_WARNING << "Clipping output idx " << idx << " of node "
                       << node->name() << " to " << output.size() - 1;
    idx = output_idx - 1;
  }
  if (idx < 0) {
    RECOM_VLOG_WARNING << "Clipping output idx " << idx << " of node "
                       << node->name() << " to " << 0;
    idx = 0;
  }
  const DataType type = output[idx].dtype();
  return type;
}

DataType FCOptimizerBase::GetTensorType(const std::string &tensor_name) {
  const NodeDef *node = node_mapping.at(GetNodeNameByTensor(tensor_name));
  const int out_idx = GetOutputIdxByTensor(tensor_name);
  return GetOutputType(node, out_idx);
}

NodeDef *FCOptimizerBase::GetInputNode(const NodeDef *node, int idx) {
  if (!node)
    return nullptr;
  return node_mapping.at(GetNodeNameByTensor(node->input(idx)));
}

int FCOptimizerBase::GetOutputTensorNum(const NodeDef *node) {
  const auto &output = properties.GetOutputProperties(node->name());
  return output.size();
}

NodeDef *FCOptimizerBase::ConstructShapeNodeByExpr(const ExprVec &shape,
                                                   DataType OutputType,
                                                   const std::string &name) {
  if (symbolic_context->IsStatic(shape)) {
    NodeDef *const_node = gd->add_node();
    const_node->set_op("Const");
    const_node->set_name(name);

    AttrValue value;
    auto tensor = value.mutable_tensor();
    tensor->set_dtype(OutputType);
    tensor->mutable_tensor_shape()->add_dim()->set_size(shape.size());
    if (OutputType == DT_INT32) {
      for (const Expression &expr : shape) {
        tensor->add_int_val(static_cast<int>(expr));
      }
    } else if (OutputType == DT_INT64) {
      for (const Expression &expr : shape) {
        tensor->add_int64_val(static_cast<int64>(expr));
      }
    } else {
      return nullptr;
    }

    (*const_node->mutable_attr())["value"] = value;
    (*const_node->mutable_attr())["dtype"].set_type(OutputType);

    return const_node;
  }

  NodeDef *shape_node = gd->add_node();
  shape_node->set_op("Addons>ShapeConstruct");
  shape_node->set_name(name);
  // shape_node->set_device(DEVICE_CPU);

  auto *mutable_attr = shape_node->mutable_attr();
  (*mutable_attr)["T"].set_type(OutputType);

  auto &exprs_attr = (*mutable_attr)["exprs"];
  for (const Expression &expr : shape) {
    exprs_attr.mutable_list()->add_s(SymEngine::str(expr));
  }

  HashSetT<std::string> retrieved;
  std::vector<std::pair<Expression, NodeDef *>> symbol_gen_node_pairs;
  for (const Expression &expr : shape) {
    auto pairs = symbolic_context->RetrieveSymbolExprGenNodePairs(expr);
    for (const auto &pair : pairs) {
      std::string key = SymEngine::str(pair.first);
      if (!retrieved.count(key)) {
        retrieved.insert(key);
        symbol_gen_node_pairs.push_back(pair);
      }
    }
  }

  auto &symbols_attr = (*mutable_attr)["symbols"];
  auto &indices_attr = (*mutable_attr)["indices"];
  auto &input_types = (*mutable_attr)["Tinputs"];
  bool all_found = true;
  for (const auto &pair : symbol_gen_node_pairs) {
    Expression symbol_expr = pair.first;
    NodeDef *generator_node = pair.second;

    bool found = false;
    const int num_generator_output =
        properties.GetOutputProperties(generator_node->name()).size();
    // generator_node->attr().at("_output_shapes").list().shape_size();
    for (int i = 0; i < num_generator_output && !found; ++i) {
      const std::string tensor_name = FormTensorName(generator_node, i);
      if (symbolic_context->ContentKnown(tensor_name)) {
        const ExprVec &content = symbolic_context->GetContent(tensor_name);
        for (int j = 0; j < content.size() && !found; ++j) {
          if (content[j] == symbol_expr) {
            shape_node->add_input(tensor_name);
            symbols_attr.mutable_list()->add_s(SymEngine::str(symbol_expr));
            indices_attr.mutable_list()->add_i(j);
            input_types.mutable_list()->add_type(
                GetOutputType(generator_node, i));
            found = true;
          }
        }
      }
    }

    for (int i = 0; i < num_generator_output && !found; ++i) {
      const std::string tensor_name = FormTensorName(generator_node, i);
      if (symbolic_context->ShapeKnown(tensor_name)) {
        const ExprVec &tensor_shape = symbolic_context->GetShape(tensor_name);
        for (int j = 0; j < tensor_shape.size() && !found; ++j) {
          if (tensor_shape[j] == symbol_expr) {
            NodeDef *generator_shape_node;
            if (!symbolic_context->HasShapeNode(tensor_name)) {
              generator_shape_node = gd->add_node();
              generator_shape_node->set_op("Shape");
              generator_shape_node->set_name(generator_node->name() +
                                             "/gen_shape");
              // generator_shape_node->set_device(DEVICE_CPU);
              (*generator_shape_node->mutable_attr())["T"].set_type(
                  GetOutputType(generator_node, i));
              (*generator_shape_node->mutable_attr())["out_type"].set_type(
                  DT_INT32);
              generator_shape_node->add_input(tensor_name);
              symbolic_context->RecordTensorShapeNode(tensor_name,
                                                      generator_shape_node);
            } else {
              generator_shape_node =
                  symbolic_context->GetShapeNode(tensor_name);
            }

            shape_node->add_input(generator_shape_node->name());
            symbols_attr.mutable_list()->add_s(SymEngine::str(symbol_expr));
            indices_attr.mutable_list()->add_i(j);
            // Cannot use get_output_type method because it may be built by us
            // after grappler shape inference
            input_types.mutable_list()->add_type(
                generator_shape_node->attr().at("out_type").type());
            found = true;
          }
        }
      }
    }

    if (!found) {
      LOG(ERROR) << "Not found match expression";
      all_found = false;
    }
  }

  (*mutable_attr)["output_dir"].set_s(output_dir);

  if (all_found)
    return shape_node;
  return nullptr;
}

} // namespace feature_opt
} // namespace tensorflow