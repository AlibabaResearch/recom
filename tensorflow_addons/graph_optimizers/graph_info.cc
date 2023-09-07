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

#include <numeric>
#include <stack>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/types.pb.h>

#include "graph_info.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

GraphInfo::GraphInfo(grappler::GrapplerItem &item) : item(item) {
  properties = std::make_unique<GraphProperties>(item);
  gd = &item.graph;

  if (UpdateAll()) {
    RECOM_VLOG << "Initialize GraphInfo successfully";
  } else {
    LOG(ERROR) << "Fail to initialize GraphInfo";
  }
}

bool GraphInfo::IsConcatOutOp(NodeDef *node) const {
  return node->op() == "ConcatV2" || node->op() == "Addons>ConcatOutputs" ||
         node->op() == "Addons>ConcatOutputsNoHost";
}

bool GraphInfo::UpdateAll() {
  if (UpdateProperties()) {
    RECOM_VLOG << "UpdateProperties succeeded";
  } else {
    LOG_AND_RETURN_FALSE("UpdateProperties failed");
  }

  if (UpdateTopoInfo()) {
    RECOM_VLOG << "UpdateTopoInfo succeeded";
  } else {
    LOG_AND_RETURN_FALSE("UpdateTopoInfo failed");
  }

  symbolic_context = std::make_shared<SymbolicShapeContext>();
  if (ReadInitConfig() && InitSymbolicShape() && SymbolicShapePropagation()) {
    RECOM_VLOG << "SymbolicShapePropagation succeeded";
  } else {
    RECOM_VLOG_WARNING << "SymbolicShapePropagation failed";
  }

  return true;
}

bool GraphInfo::UpdateProperties() {
  // properties->Clear();
  for (NodeDef &node : *(gd->mutable_node())) {
    properties->ClearInputProperties(node.name());
    properties->ClearOutputProperties(node.name());
  }
  RETURN_IF_FALSE(properties->InferStatically(true, true, true) ==
                  Status::OK());
  RETURN_IF_FALSE(properties->AnnotateOutputShapes(gd) == Status::OK());
  return true;
}

bool GraphInfo::ReadInitConfig() {
  // TODO: pass the placeholder
  return true;
}

bool GraphInfo::InitSymbolicShape() {
  // TODO: init by config
  for (NodeDef &node : *(gd->mutable_node())) {
    // init placeholder shape with batch-size
    if (node.op() == "Placeholder") {
      auto shape = node.attr().at("shape").shape();

      ExprVec placeholder_shape(shape.dim_size());
      int num_elements = 1;
      bool is_dynamic = false;
      for (int i = 0; i < shape.dim_size(); i++) {
        int dim = shape.dim(i).size();
        if (dim == -1) {
          placeholder_shape[i] = symbolic_context->AddNewSymbol(&node);
          is_dynamic = true;
        } else {
          placeholder_shape[i] = Expression(dim);
          num_elements *= dim;
        }
      }

      symbolic_context->SetShape(node.name(), placeholder_shape);

      if (!is_dynamic) {
        ExprVec content(num_elements);
        for (int i = 0; i < num_elements; ++i) {
          content[i] = symbolic_context->AddNewSymbol(&node);
        }
        symbolic_context->SetContent(node.name(), content);
      }
    }
  }

  return true;
}

bool GraphInfo::SymbolicShapePropagation() {
  HashMapT<string, int> node_indegrees = indegree_mapping;
  std::stack<NodeDef *> node_stack;

  bool all_success = true;
  for (int i = 0; i < fc_node_sets.size(); ++i) {
    std::stack<NodeDef *> node_stack;
    for (const std::string &node_name : fc_node_sets[i]) {
      if (node_indegrees.at(node_name) == 0)
        node_stack.push(node_mapping.at(node_name));
    }

    bool success = true;
    while (!node_stack.empty()) {
      NodeDef *node = node_stack.top();
      node_stack.pop();
      if (node->op() != "NoOp" && node->op() != "Placeholder") {
        if (!RunSymbolicFn(symbolic_context, node)) {
          RECOM_VLOG_WARNING << "Fail to handle " << node->name()
                             << " op: " << node->op();
          success = false;
        }
      }

      for (const std::string &out_name : out_mapping.at(node->name())) {
        if (--node_indegrees[out_name] == 0)
          node_stack.push(node_mapping.at(out_name));
      }
    }

    all_success = all_success && success;
  }

  return all_success;
}

bool GraphInfo::UpdateTopoInfo() {
  node_mapping = {};
  out_mapping = {};
  for (NodeDef &node : *gd->mutable_node()) {
    std::string node_name = node.name();
    node_mapping.emplace(node_name, &node);

    int input_num = node.input_size();
    for (int i = 0; i < input_num; i++) {
      std::string prenode_name = GetNodeNameByTensor(node.input(i));
      out_mapping[prenode_name].insert(node_name);
    }
  }

  for (NodeDef &node : *gd->mutable_node()) {
    if (!out_mapping.count(node.name())) {
      out_mapping[node.name()] = {};
    }
  }

  indegree_mapping = {};
  for (NodeDef &node : *gd->mutable_node()) {
    HashSetT<std::string> input_node_names;
    const int num_input = node.input_size();
    for (int i = 0; i < num_input; ++i) {
      input_node_names.insert(GetNodeNameByTensor(node.input(i)));
    }
    indegree_mapping[node.name()] = input_node_names.size();
  }

  std::stack<NodeDef *> node_stack;
  for (const auto &name_indegree : indegree_mapping) {
    if (name_indegree.second == 0) {
      NodeDef *node = node_mapping.at(name_indegree.first);
      if (node->op() != "Placeholder")
        node_stack.push(node);
    }
  }

  nonconst_indegree_mapping = indegree_mapping;
  while (!node_stack.empty()) {
    NodeDef *node = node_stack.top();
    node_stack.pop();

    for (const std::string &out_name : out_mapping.at(node->name())) {
      if (--nonconst_indegree_mapping.at(out_name) == 0) {
        node_stack.push(node_mapping.at(out_name));
      }
    }
  }

  RETURN_IF_FALSE(ExtractFCNodes());

  return true;
}

bool GraphInfo::ExtractFCNodes() {
  std::vector<NodeDef *> embedding_tables;
  for (NodeDef &variable : *gd->mutable_node()) {
    // TODO: support resource
    if (variable.op() == "VariableV2") {
      auto is_embedding_table = [&]() -> bool {
        std::stack<NodeDef *> node_stack;
        for (const std::string &out_name : out_mapping.at(variable.name())) {
          NodeDef *out_node = node_mapping.at(out_name);
          node_stack.push(out_node);
        }

        int cnt = 0;
        while (!node_stack.empty()) {
          NodeDef *node = node_stack.top();
          node_stack.pop();

          if (node->op() == "Identity") {
            for (const std::string &out_name : out_mapping.at(node->name())) {
              NodeDef *out_node = node_mapping.at(out_name);
              node_stack.push(out_node);
            }
          } else if (node->op() == "Assign" || node->op() == "SaveV2") {
            // pass
          } else if (node->op().find("Gather") == std::string::npos &&
                     node->op().find("SparseSegment") == std::string::npos) {
            if (variable.name().find("embedding_weights") !=
                std::string::npos) {
              RECOM_VLOG_WARNING << "Find non-embedding variable "
                                 << variable.name() << " successor node "
                                 << node->name() << " op " << node->op();
            }
            return false;
          } else {
            ++cnt;
          }
        }

        return cnt > 0;
      };

      if (is_embedding_table()) {
        embedding_tables.push_back(&variable);
        if (variable.name().find("embedding_weights") == std::string::npos) {
          RECOM_VLOG_WARNING
              << "Find embedding table not ends with embedding_weights "
              << variable.name();
        }
      }
    }
  }

  HashMapT<std::string, HashSetT<NodeDef *>> pre_tables_mapping;
  for (NodeDef *table : embedding_tables) {
    pre_tables_mapping[table->name()] = {table};
  }

  std::stack<NodeDef *> node_stack;
  HashMapT<std::string, int> left_indegrees = indegree_mapping;
  for (const auto &name_indegree : left_indegrees) {
    if (name_indegree.second == 0)
      node_stack.push(node_mapping.at(name_indegree.first));
  }

  while (!node_stack.empty()) {
    NodeDef *node = node_stack.top();
    node_stack.pop();

    if (!pre_tables_mapping.count(node->name())) {
      HashSetT<NodeDef *> pre_tables;
      HashSetT<std::string> visited;
      if (node->op() != "Assign" && node->op() != "SaveV2") {
        for (const std::string &input_tensor : node->input()) {
          std::string inode_name = GetNodeNameByTensor(input_tensor);
          if (!visited.count(inode_name)) {
            HashSetT<NodeDef *> inode_pre_tables =
                pre_tables_mapping.at(inode_name);
            pre_tables.insert(inode_pre_tables.begin(), inode_pre_tables.end());
            visited.insert(inode_name);
          }
        }
      }
      pre_tables_mapping[node->name()] = pre_tables;
    }

    for (const std::string &out_name : out_mapping.at(node->name())) {
      NodeDef *out_node = node_mapping.at(out_name);
      if (--left_indegrees.at(out_name) == 0) {
        node_stack.push(out_node);
      }
    }
  }

  fc_node_sets = {};
  fc_boundary_node_sets = {};
  for (NodeDef *table : embedding_tables) {
    HashSetT<std::string> fc_boundary_node_set;
    std::stack<NodeDef *> node_stack;
    node_stack.push(table);
    while (!node_stack.empty()) {
      NodeDef *node = node_stack.top();
      node_stack.pop();

      for (const std::string &out_name : out_mapping.at(node->name())) {
        NodeDef *out_node = node_mapping.at(out_name);
        int num_onode_pre_tables = pre_tables_mapping.at(out_name).size();
        if (num_onode_pre_tables <= 1) {
          node_stack.push(out_node);
        } else {
          fc_boundary_node_set.insert(node->name());
          if (!IsConcatOutOp(out_node)) {
            RECOM_VLOG_WARNING
                << "Find boundary out node not concat " << out_name << " op "
                << out_node->op() << " #pre_table " << num_onode_pre_tables
                << " boundary_node " << node->name() << " op " << node->op();
          }
        }
      }
    }

    HashSetT<std::string> fc_node_set;
    fc_node_set.insert(fc_boundary_node_set.begin(),
                       fc_boundary_node_set.end());

    for (const std::string &boundary_name : fc_boundary_node_set) {
      node_stack.push(node_mapping.at(boundary_name));
    }
    while (!node_stack.empty()) {
      NodeDef *node = node_stack.top();
      node_stack.pop();

      for (int i = 0; i < node->input_size(); ++i) {
        const std::string iname = GetNodeNameByTensor(node->input(i));
        if (!fc_node_set.count(iname)) {
          node_stack.push(node_mapping.at(iname));
          fc_node_set.insert(iname);
        }
      }
    }

    if (fc_node_set.size() > 0) {
      RECOM_VLOG << "Extract FC " << fc_node_sets.size() << " with "
                 << fc_node_set.size() << " nodes";

      fc_node_sets.push_back(fc_node_set);
      fc_boundary_node_sets.push_back(fc_boundary_node_set);
    }
  }

  int total_nodes = 0;
  for (const auto &fc_node_set : fc_node_sets) {
    total_nodes += fc_node_set.size();
  }
  RECOM_VLOG << "FC total nodes " << total_nodes;

  return true;
}

bool GraphInfo::PruneFCDeadNodes() {
  HashSetT<NodeDef *> dead_nodes;
  for (int i = 0; i < fc_node_sets.size(); ++i) {
    HashMapT<string, int> left_indegrees = nonconst_indegree_mapping;
    std::stack<NodeDef *> node_stack;
    for (const std::string &node_name : fc_node_sets[i]) {
      NodeDef *node = node_mapping.at(node_name);
      if (node->op() == "Placeholder") {
        node_stack.push(node);
        break;
      }
    }

    HashSetT<NodeDef *> candidates;
    while (!node_stack.empty()) {
      NodeDef *node = node_stack.top();
      node_stack.pop();
      candidates.insert(node);

      for (const std::string &out_name : out_mapping.at(node->name())) {
        NodeDef *out_node = node_mapping.at(out_name);
        if (--left_indegrees.at(out_name) == 0) {
          node_stack.push(out_node);
        }
      }
    }

    while (true) {
      HashSetT<NodeDef *> updated = candidates;
      for (NodeDef *node : candidates) {
        for (int i = 0; i < node->input_size(); ++i) {
          const std::string iname = GetNodeNameByTensor(node->input(i));
          NodeDef *inode = node_mapping.at(iname);
          if (!candidates.count(inode)) {
            updated.insert(inode);
          }
        }
      }

      if (updated.size() != candidates.size()) {
        candidates = updated;
      } else {
        break;
      }
    }

    HashSetT<NodeDef *> fc_dead_nodes;
    int last_size;
    do {
      last_size = fc_dead_nodes.size();
      for (NodeDef *candidate : candidates) {
        if (out_mapping.at(candidate->name()).empty()) {
          fc_dead_nodes.insert(candidate);
          for (int i = 0; i < candidate->input_size(); ++i) {
            out_mapping[GetNodeNameByTensor(candidate->input(i))].erase(
                candidate->name());
          }
        }
      }
    } while (fc_dead_nodes.size() > last_size);

    dead_nodes.insert(fc_dead_nodes.begin(), fc_dead_nodes.end());

    RECOM_VLOG << "Got " << fc_dead_nodes.size() << " dead nodes for FC " << i;
  }

  GraphDef new_gd;
  for (NodeDef &node : *gd->mutable_node()) {
    if (!dead_nodes.count(&node)) {
      NodeDef *new_node = new_gd.add_node();
      new_node->CopyFrom(node);
    }
  }
  gd->Swap(&new_gd);

  RECOM_VLOG << "Prune " << dead_nodes.size() << " dead nodes";

  return true;
}

bool GraphInfo::RenameFCNodes() {
  UpdateTopoInfo();
  HashMapT<std::string, std::string> new_name_mapping;
  for (int i = 0; i < fc_node_sets.size(); ++i) {
    std::string name_prefix;
    for (const std::string &node_name : fc_node_sets[i]) {
      NodeDef *node = node_mapping.at(node_name);
      if (node->op() == "Addons>ExtendedSparseSegmentMean" ||
          node->op() == "Addons>ExtendedSparseSegmentSum") {
        name_prefix = GetEmbeddingNamePrefix(node_name);
        break;
      }
    }

    if (name_prefix != "") {
      HashMapT<std::string, int> name_counters;
      for (const std::string &node_name : fc_node_sets[i]) {
        if (GetEmbeddingNamePrefix(node_name) == name_prefix) {
          NodeDef *node = node_mapping.at(node_name);
          std::string dominant_name = name_prefix + "/" + node->op();

          int cnt = name_counters[dominant_name]++;
          std::string suffix = "_";
          do {
            suffix.push_back(cnt % 26 + 'a');
            cnt /= 26;
          } while (cnt > 0);

          std::string new_name = dominant_name + suffix;
          node->set_name(new_name);
          new_name_mapping[node_name] = new_name;
        }
      }
    }
  }

  for (NodeDef &node : *gd->mutable_node()) {
    for (int i = 0; i < node.input_size(); ++i) {
      const std::string input_name = GetNodeNameByTensor(node.input(i));
      const int idx = GetOutputIdxByTensor(node.input(i));
      if (new_name_mapping.count(input_name)) {
        const std::string new_name = new_name_mapping.at(input_name);
        *node.mutable_input(i) =
            idx == 0 ? new_name : (new_name + ":" + std::to_string(idx));
      }
    }
  }

  return true;
}

} // namespace feature_opt
} // namespace tensorflow
