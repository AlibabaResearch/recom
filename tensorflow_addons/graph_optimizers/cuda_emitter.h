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
#include <queue>
#include <string>
#include <vector>

#include "fc_optimizer_base.h"

namespace tensorflow {
namespace feature_opt {

class CudaEmitter : public FCOptimizerBase {
  struct Variable {
    std::string name;
    std::string type;

    Variable() = default;
    Variable(const std::string &name, const std::string &type)
        : name(name), type(type) {}

    std::string NormalDeclare() const { return type + " " + name; }

    std::string ConstNormalDeclare() const {
      return "const " + type + " " + name;
    }

    std::string PtrDeclare() const { return type + " *" + name; }

    std::string RefParam() const { return type + " &" + name; }

    std::string ConstRefParam() const { return "const " + type + " &" + name; }

    std::string ConstRestrictPtrParam() const {
      return "const " + type + " *__restrict__ " + name;
    }

    std::string RestrictPtrParam() const {
      return type + " *__restrict__ " + name;
    }

    std::string PtrRefParam() const { return type + " *&" + name; }

    std::string At(int idx) const {
      return name + "[" + std::to_string(idx) + "]";
    }

    std::string At(const std::string &idx_str) const {
      return name + "[" + idx_str + "]";
    }
  };

  struct ArrayVariable : public Variable {
    int size;

    ArrayVariable() = default;
    ArrayVariable(const std::string &name, const std::string &type, int size)
        : Variable(name, type), size(size) {}

    std::string ArrDeclare() const {
      return type + " " + name + "[" + std::to_string(size) + "]";
    }

    std::string ArrRefParam() const {
      return type + " (&" + name + ")[" + std::to_string(size) + "]";
    }
  };

  struct FCMeta {
    // tensor_tuple: (tensor_name, dtype, rank)
    std::vector<std::tuple<std::string, DataType, int>>
        device_input_tensor_tuples;
    std::vector<std::tuple<std::string, DataType, int>>
        host_input_tensor_tuples;
    // var_tuple: (var, shape_var, shape_exprs)
    std::vector<std::tuple<Variable, ArrayVariable, ExprVec>>
        device_input_var_tuples;
    std::vector<std::tuple<Variable, ArrayVariable, ExprVec>>
        host_input_var_tuples;

    HashSetT<std::string> input_tensors;
    std::vector<Variable> input_vars;
    HashSetT<std::string> input_symbol_strs;

    ExprVec shape_inputs;

    std::vector<std::tuple<std::string, DataType, int>> output_tensor_tuples;
    std::vector<Variable> real_output_vars;
    HashMapT<std::string, Variable> value_output_real_var_mapping;
    HashMapT<std::string, Variable> real_output_var_mapping;
    HashMapT<std::string, ExprVec> real_output_var_symbolic_shape_mapping;
    std::vector<NodeDef *> value_output_nodes;

    std::vector<std::pair<Variable, Expression>> symbol_upper_bounds;
    std::vector<std::pair<Variable, Expression>> buffers; // upper bound
    std::vector<std::pair<Variable, std::vector<std::string>>> const_buffers;
  };

  struct SubgraphCode {
    std::string pre_glb_area;
    std::string loop_body;
    std::string post_glb_area;

    HashMapT<std::string, Variable> inputs;
    HashMapT<std::string, Variable> outputs;
    HashMapT<std::string, ArrayVariable> shared_params;
  };

  // Graph Meta
  NodeDef *symbols_input_node;

  const uint64 max_table_size;
  const int block_threads;
  const std::string block_threads_str;

  HashMapT<std::string, int> node_id_mapping;

public:
  CudaEmitter(GraphInfo &ginfo, uint64 max_table_size, int block_threads)
      : FCOptimizerBase(ginfo), symbols_input_node(nullptr),
        max_table_size(max_table_size), block_threads(block_threads),
        block_threads_str(std::to_string(block_threads)) {
    int id = 0;
    for (const auto &node_pair : node_mapping) {
      node_id_mapping[node_pair.first] = id++;
    }
  }

  bool Optimize();

private:
  std::string MakeTensorVarName(const std::string &tensor_name);
  std::string MakeTensorVarName(const NodeDef *node, int idx = 0);
  std::string MakeTensorShapeVarName(const std::string &tensor_name);
  std::string MakeTensorShapeVarName(const NodeDef *node, int idx = 0);

  bool IsReshape(const NodeDef *node, const int out_idx, int &in_idx);
  bool IsReshape(const NodeDef *node, const int out_idx);

  bool EmitCodes(std::vector<std::shared_ptr<FCMeta>> &fc_metas,
                 std::string &code, std::vector<bool> &success_flags);

  bool SetFCBeginToCPU(NodeDef *kernel_inode);

  bool SetUnemitFCToCPU(const HashSetT<std::string> &fc_node_set,
                        const HashSetT<std::string> &fc_boundary_node_set);

  bool EmitHeaders(std::string &headers);

  bool EmitFCCode(const HashSetT<std::string> &fc_node_set,
                  const HashSetT<std::string> &fc_boundary_node_set, int fc_id,
                  FCMeta &fc_meta, std::string &fc_code_string);

  bool FindFCOutputs(const HashSetT<std::string> &fc_boundary_node_set,
                     FCMeta &fc_meta);

  bool EmitSubgraphCode(NodeDef *node, int port, FCMeta &fc_meta,
                        SubgraphCode &code,
                        std::queue<std::pair<NodeDef *, int>> &node_queue,
                        bool &is_input);

  bool EmitBatchColReduction(NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
                             std::queue<std::pair<NodeDef *, int>> &node_queue);

  bool EmitGatherRows(NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
                      std::queue<std::pair<NodeDef *, int>> &node_queue);

  bool EmitGatherScatterRows(NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
                             std::queue<std::pair<NodeDef *, int>> &node_queue);

  bool
  EmitSparseSegmentReduce(NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
                          std::queue<std::pair<NodeDef *, int>> &node_queue);

  bool EmitSparseSegmentReduceExperiment(
      NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
      std::queue<std::pair<NodeDef *, int>> &node_queue);

  bool EmitElementWise(NodeDef *node, FCMeta &fc_meta, SubgraphCode &code,
                       std::queue<std::pair<NodeDef *, int>> &node_queue);

  bool EmitInputInline(NodeDef *node,
                       const std::pair<Expression, Expression> &index_expr_pair,
                       int out_idx, FCMeta &fc_meta, SubgraphCode &code,
                       std::string &inline_code,
                       std::queue<std::pair<NodeDef *, int>> &node_queue);

  bool ConstructSubgraphCode(const SubgraphCode &subgraph_code, int unique_id,
                             std::string &subgraph_code_string);

  bool
  ConstructFCCode(FCMeta &fc_meta,
                  std::vector<std::shared_ptr<SubgraphCode>> &subgraph_codes,
                  int fc_id, std::string &fc_code_string);

  bool
  ConstructKernelEntry(const std::vector<std::shared_ptr<FCMeta>> &fc_metas,
                       std::string &kernel);

  bool
  ConstructKernelCaller(const std::vector<std::shared_ptr<FCMeta>> &fc_metas,
                        std::string &host);

  bool ConstructConstBufferPrepareEntry(
      const std::vector<std::shared_ptr<FCMeta>> &fc_metas,
      std::string &code_string);

  bool
  ConstructOpComputeEntry(const std::vector<std::shared_ptr<FCMeta>> &fc_metas,
                          std::string &code_string);

  bool Rewrite(const std::vector<std::shared_ptr<FCMeta>> &fc_metas,
               const std::string &dlpath);
};

} // namespace feature_opt
} // namespace tensorflow