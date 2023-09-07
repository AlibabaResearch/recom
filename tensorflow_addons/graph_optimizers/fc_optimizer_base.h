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
#include <symengine/expression.h>
#include <tensorflow/core/framework/types.pb.h>
#include <vector>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

#include "arithm_binary_op_mapping.h"
#include "graph_info.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn_registry.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

class FCOptimizerBase {
protected:
  GraphDef *gd;

  OutMap &out_mapping;
  NodeMap &node_mapping;
  std::vector<HashSetT<std::string>> &fc_node_sets;
  std::vector<HashSetT<std::string>> &fc_boundary_node_sets;

  const GraphProperties &properties;

  ArithmBinaryOpMapping arithm_mapping;

  std::shared_ptr<SymbolicShapeContext> symbolic_context;

  std::string output_dir;

public:
  FCOptimizerBase(GraphInfo &ginfo)
      : gd(ginfo.gd), out_mapping(ginfo.out_mapping),
        node_mapping(ginfo.node_mapping), fc_node_sets(ginfo.fc_node_sets),
        fc_boundary_node_sets(ginfo.fc_boundary_node_sets),
        properties(*ginfo.properties.get()),
        symbolic_context(ginfo.symbolic_context),
        output_dir(GetEnv("RECOM_CACHE_DIR", "/tmp/RECOM")) {
    if (!ExistFile(output_dir)) {
      RECOM_VLOG << "mkdir " << output_dir;
      if (system(("mkdir -p " + output_dir).c_str()) != 0) {
        LOG(ERROR) << "mkdir failed";
      }
    }
  }

  virtual ~FCOptimizerBase() = default;

protected:
  NodeDef *FindOutputNode(const NodeDef *node, const std::string &op_types);

  NodeDef *FindInputNodeByIdx(const NodeDef *node, const int idx,
                              const std::string &op_types);

  DataType GetInputType(const NodeDef *node, const int input_idx);

  DataType GetOutputType(const NodeDef *node, const int output_idx);

  DataType GetTensorType(const std::string &tensor_name);

  NodeDef *GetInputNode(const NodeDef *node, int idx);

  int GetOutputTensorNum(const NodeDef *node);

  NodeDef *ConstructShapeNodeByExpr(const ExprVec &content, DataType OutputType,
                                    const std::string &name);

  template <typename T> bool ExtractConst(NodeDef *node, T &c) {
    std::vector<T> vec;
    RETURN_IF_FALSE(ExtractConstTensor(node, vec));
    RETURN_IF_FALSE(vec.size() == 1);
    c = vec[0];

    return true;
  }

  template <typename SrcT, typename DstT>
  std::vector<DstT> ConvertTensorToVec(const Tensor &tensor) {
    const SrcT *data = tensor.flat<SrcT>().data();
    const int num_elements = tensor.NumElements();
    std::vector<DstT> vec(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      vec[i] = static_cast<DstT>(data[i]);
    }
    return vec;
  }

  template <typename T>
  bool ExtractConstTensor(NodeDef *node, std::vector<T> &vec) {
    // TODO: utilize the Grappler constant folding
    RETURN_IF_FALSE(node);

    if (symbolic_context->ContentStatic(node->name())) {
      const ExprVec content = symbolic_context->GetContent(node->name());
      vec = std::vector<T>(content.size());
      for (int i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<T>(content[i]);
      }
    } else if (node->op() == "Const") {
      Tensor tensor;
      LOG_AND_RETURN_IF_FALSE(
          tensor.FromProto(node->attr().at("value").tensor()),
          "Convert TensorProto of Const op to Tensor fail");

      switch (tensor.dtype()) {
      case DT_INT32:
        vec = ConvertTensorToVec<int, T>(tensor);
        break;
      case DT_INT64:
        vec = ConvertTensorToVec<int64, T>(tensor);
        break;
      case DT_FLOAT:
        vec = ConvertTensorToVec<float, T>(tensor);
        break;
      case DT_DOUBLE:
        vec = ConvertTensorToVec<double, T>(tensor);
        break;
      case DT_STRING:
        RECOM_VLOG_WARNING << "Unsupported type of Const: tstring";
        RETURN_FALSE;
      default:
        // TODO: handle more type
        RECOM_VLOG_WARNING << "Currently do not support the type of Const "
                           << DataTypeString(tensor.dtype());
        RETURN_FALSE;
      }
    } else if (node->op() == "Cast") {
      return ExtractConstTensor(GetInputNode(node, 0), vec);
    } else if (arithm_mapping.count<T>(node->op())) {
      // TODO: consider accuracy loss caused by cast
      std::vector<T> vec1, vec2;
      RETURN_IF_FALSE(ExtractConstTensor(GetInputNode(node, 0), vec1) &&
                      ExtractConstTensor(GetInputNode(node, 1), vec2));
      LOG_AND_RETURN_IF_FALSE(
          vec1.size() == vec2.size(),
          "Currently not support const Tensor broadcast computing");
      vec = std::vector<T>(vec1.size());
      for (int i = 0; i < vec.size(); ++i) {
        vec[i] = arithm_mapping.map<T>(node->op())(vec1[i], vec2[i]);
      }
    } else {
      return false;
    }

    return true;
  }

  template <typename T> bool ExtractSplatConst(NodeDef *node, T &c) {
    RETURN_IF_FALSE(node);

    if (node->op() == "ZerosLike") {
      c = 0;
    } else if (node->op() == "OnesLike") {
      c = 1;
    } else if (node->op() == "Fill") {
      return ExtractConst(GetInputNode(node, 1), c);
    } else if (arithm_mapping.count<T>(node->op())) {
      T c1, c2;
      RETURN_IF_FALSE(ExtractSplatConst(GetInputNode(node, 0), c1) &&
                      ExtractSplatConst(GetInputNode(node, 1), c2));
      c = arithm_mapping.map<T>(node->op())(c1, c2);
    } else if (node->op() == "Reshape" || node->op() == "ExpandDims" ||
               node->op() == "Squeeze") {
      return ExtractSplatConst(GetInputNode(node, 0), c);
    } else if (node->op() == "Tile") {
      return ExtractSplatConst(GetInputNode(node, 0), c);
    } else if (node->op() == "Cast") {
      return ExtractSplatConst(GetInputNode(node, 0), c);
    } else if (node->op() == "StridedSlice" || node->op() == "Slice") {
      return ExtractSplatConst(GetInputNode(node, 0), c);
    } else if (ExtractConst(node, c)) {
    } else {
      RETURN_FALSE;
    }

    return true;
  }

#define CONVERGE_OR_RETURN(NODE1, NODE2)                                       \
  if (NODE1 != NODE2) {                                                        \
    return false;                                                              \
  }

#define FIND_INPUT_OR_RETURN(NODE, INODE, IDX, OP_TYPES)                       \
  NodeDef *INODE = FindInputNodeByIdx(NODE, IDX, OP_TYPES);                    \
  if (!INODE)                                                                  \
    return false;

#define FIND_OUTPUT_OR_RETURN(NODE, ONODE, OP_TYPES)                           \
  NodeDef *ONODE = FindOutputNode(NODE, OP_TYPES);                             \
  if (!ONODE)                                                                  \
    return false;

#define EXTRACT_CONST_OR_RETURN(NODE, c, T)                                    \
  T c;                                                                         \
  RETURN_IF_FALSE(ExtractConst((NODE), c));

#define INPUT_IS_CONST_OR_RETURN(NODE, IDX, VALUE, T)                          \
  {                                                                            \
    EXTRACT_CONST_OR_RETURN(GetInputNode((NODE), (IDX)), c, T);                \
    RETURN_IF_FALSE(c == VALUE);                                               \
  }

#define EXTRACT_SPLAT_CONST_OR_RETURN(NODE, c, T)                              \
  T c;                                                                         \
  RETURN_IF_FALSE(ExtractSplatConst((NODE), c));

#define INPUT_IS_SPLAT_CONST_OR_RETURN(NODE, IDX, VALUE, T)                    \
  {                                                                            \
    EXTRACT_SPLAT_CONST_OR_RETURN(GetInputNode((NODE), (IDX)), c, T);          \
    RETURN_IF_FALSE(c == VALUE);                                               \
  }

#define EXTRACT_CONST_TENSOR_OR_RETURN(NODE, c, T)                             \
  std::vector<T> c;                                                            \
  RETURN_IF_FALSE(ExtractConstTensor((NODE), c));

#define INPUT_IS_CONST_TENSOR_OR_RETURN(NODE, IDX, VALUE, T)                   \
  {                                                                            \
    EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode((NODE), (IDX)), c, T)          \
    std::vector<T> val(VALUE);                                                 \
    RETURN_IF_FALSE(c == val);                                                 \
  }

#define FIND_BI_COMMUT_OP_INPUT_OR_RETURN(NODE, INODE, IDX, OP_TYPES)          \
  NodeDef *INODE;                                                              \
  int IDX;                                                                     \
  for (IDX = 0; IDX < 2; ++IDX) {                                              \
    INODE = FindInputNodeByIdx(NODE, IDX, OP_TYPES);                           \
    if (INODE)                                                                 \
      break;                                                                   \
  }                                                                            \
  if (!INODE)                                                                  \
    return false;
};

} // namespace feature_opt
} // namespace tensorflow