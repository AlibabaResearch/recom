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

#include "pre_lookup_optimizer.h"
#include "tensorflow_addons/utils.h"
#include <tensorflow/core/framework/types.pb.h>
#include <unordered_map>
#include <vector>

namespace tensorflow {
namespace feature_opt {

const PreLookupOptimizer::IntervalSet PreLookupOptimizer::UniversalSet =
    PreLookupOptimizer::IntervalSet(
        PreLookupOptimizer::Interval::closed(INT_MIN, INT_MAX));

void PreLookupOptimizer::Optimize() {
  for (int i = 0; i < fc_node_sets.size(); ++i) {
    for (const std::string &node_name : fc_node_sets[i]) {
      NodeDef *node = node_mapping.at(node_name);
      if (node->op() == "SparseFillEmptyRows") {
        if (Optimize(node)) {
          RECOM_VLOG << "PreLookupOptimizer: optimize FC " << i
                     << " successfully";
        } else {
          RECOM_VLOG << "PreLookupOptimizer: fail to optimize FC " << i;
        }
        break;
      }
    }
  }
}

bool PreLookupOptimizer::Optimize(NodeDef *sfer) {
  indice_reserve_ops = std::stack<NodeDef *>();
  value_reserve_ops = std::stack<NodeDef *>();
  meta_vec = std::vector<OpMeta>();

  NodeDef *indice_node = GetInputNode(sfer, 0);
  NodeDef *value_node = GetInputNode(sfer, 1);
  int indice_out_idx = GetOutputIdxByTensor(sfer->input(0));
  int value_out_idx = GetOutputIdxByTensor(sfer->input(1));
  DataType init_indice_type;
  RETURN_IF_FALSE(MatchBeforeLookup(value_node, value_out_idx, indice_node,
                                    indice_out_idx, init_indice_type));
  RECOM_VLOG << "Match Sucessfully";
  RETURN_IF_FALSE(Simplify());
  RECOM_VLOG << "Simplify Sucessfully";
  RETURN_IF_FALSE(ReconstructGraph(sfer, value_node, value_out_idx, indice_node,
                                   indice_out_idx, init_indice_type));
  RECOM_VLOG << "Reconstruct Sucessfully";

  // TODO: prune dead nodes

  return true;
}

bool PreLookupOptimizer::MatchExpr(const NodeDef *op_node,
                                   NodeDef *&in_value_node,
                                   int &in_value_out_idx, IntervalSet &s) {
  RETURN_IF_FALSE(op_node);

  // RECOM_VLOG << "op_node: [" << op_node->name() << ", " << op_node->op() <<
  // "]";

  if (op_node->op() == "LogicalOr" || op_node->op() == "LogicalAnd") {
    NodeDef *left = GetInputNode(op_node, 0);
    NodeDef *right = GetInputNode(op_node, 1);
    IntervalSet ls, rs;
    RETURN_IF_FALSE(MatchExpr(left, in_value_node, in_value_out_idx, ls) &&
                    MatchExpr(right, in_value_node, in_value_out_idx, rs));

    if (op_node->op() == "LogicalOr") {
      s = ls | rs;
    } else if (op_node->op() == "LogicalAnd") {
      s = ls & rs;
    }
  } else if (op_node->op() == "LogicalNot") {
    IntervalSet ns;
    RETURN_IF_FALSE(MatchExpr(GetInputNode(op_node, 0), in_value_node,
                              in_value_out_idx, ns));
    s = UniversalSet - ns;
  } else if (op_node->op() == "GreaterEqual" || op_node->op() == "LessEqual" ||
             op_node->op() == "Greater" || op_node->op() == "Less" ||
             op_node->op() == "NotEqual") {
    NodeDef *in_value_x1 = GetInputNode(op_node, 0);
    int in_value_out_idx_x1 = GetOutputIdxByTensor(op_node->input(0));
    if (in_value_node) {
      CONVERGE_OR_RETURN(in_value_node, in_value_x1);
      RETURN_IF_FALSE(in_value_out_idx == in_value_out_idx_x1);
    } else {
      in_value_node = in_value_x1;
      in_value_out_idx = in_value_out_idx_x1;
    }

    NodeDef *x = GetInputNode(op_node, 1);
    EXTRACT_SPLAT_CONST_OR_RETURN(x, c, int);

    if (op_node->op() == "GreaterEqual") {
      s = IntervalSet(Interval::closed(c, INT_MAX));
    } else if (op_node->op() == "Greater") {
      s = IntervalSet(Interval::left_open(c, INT_MAX));
    } else if (op_node->op() == "LessEqual") {
      s = IntervalSet(Interval::closed(INT_MIN, c));
    } else if (op_node->op() == "Less") {
      s = IntervalSet(Interval::right_open(INT_MIN, c));
    } else if (op_node->op() == "NotEqual") {
      s = UniversalSet - c;
    }
  } else {
    RETURN_FALSE;
  }

  return true;
}

bool PreLookupOptimizer::MatchGatherValue(NodeDef *&value_node,
                                          int &value_out_idx,
                                          NodeDef *&indice_node,
                                          int &indice_out_idx) {
  RETURN_IF_FALSE(value_node);

  if (!indice_node)
    return MatchGatherValue(value_node, value_out_idx);

  NodeDef *where, *in_value_node, *in_indice_node;
  int in_value_out_idx, in_indice_out_idx;
  if (value_node->op() == "GatherV2") {
    in_value_node = GetInputNode(value_node, 0);
    in_value_out_idx = GetOutputIdxByTensor(value_node->input(0));
    FIND_INPUT_OR_RETURN(value_node, reshape, 1, "Reshape");
    INPUT_IS_CONST_OR_RETURN(value_node, 2, 0, int);
    FIND_INPUT_OR_RETURN(reshape, where_tmp, 0, "Where");
    where = where_tmp;

    RETURN_IF_FALSE(indice_node->op() == "GatherV2");

    in_indice_node = GetInputNode(indice_node, 0);
    in_indice_out_idx = GetOutputIdxByTensor(indice_node->input(0));
    // TODO: match reshape
    FIND_INPUT_OR_RETURN(indice_node, reshape_x1, 1, "Reshape");
    CONVERGE_OR_RETURN(reshape, reshape_x1);
    INPUT_IS_CONST_OR_RETURN(indice_node, 2, 0, int);
  } else if (value_node->op() == "GatherNd") {
    in_value_node = GetInputNode(value_node, 0);
    in_value_out_idx = GetOutputIdxByTensor(value_node->input(0));
    FIND_INPUT_OR_RETURN(value_node, where_tmp, 1, "Where");
    where = where_tmp;

    RETURN_IF_FALSE(indice_node->op() == "Where");
    CONVERGE_OR_RETURN(indice_node, where);
    in_indice_node = nullptr; // indice is generated from Where
    in_indice_out_idx = 0;
  } else {
    RETURN_FALSE;
  }

  IntervalSet s;
  RETURN_IF_FALSE(
      MatchExpr(GetInputNode(where, 0), in_value_node, in_value_out_idx, s));

  value_node = in_value_node;
  value_out_idx = in_value_out_idx;
  indice_node = in_indice_node;
  indice_out_idx = in_indice_out_idx;
  meta_vec.push_back(OpMeta(OpMeta::Type::Gather, s));
  return true;
}

bool PreLookupOptimizer::MatchGatherValue(NodeDef *&value_node,
                                          int &value_out_idx) {
  RETURN_IF_FALSE(value_node);

  NodeDef *where, *in_value_node;
  int in_value_out_idx;
  if (value_node->op() == "GatherV2") {
    in_value_node = GetInputNode(value_node, 0);
    in_value_out_idx = GetOutputIdxByTensor(value_node->input(0));
    FIND_INPUT_OR_RETURN(value_node, reshape, 1, "Reshape");
    FIND_INPUT_OR_RETURN(value_node, value_axis, 2, "Const");
    INPUT_IS_CONST_OR_RETURN(value_node, 2, 0, int);
    FIND_INPUT_OR_RETURN(reshape, where_tmp, 0, "Where");
    where = where_tmp;
  } else if (value_node->op() == "GatherNd") {
    FIND_INPUT_OR_RETURN(value_node, where_tmp, 1, "Where");
    where = where_tmp;
  } else {
    RETURN_FALSE;
  }

  IntervalSet s;
  RETURN_IF_FALSE(
      MatchExpr(GetInputNode(where, 0), in_value_node, in_value_out_idx, s));

  value_node = in_value_node;
  value_out_idx = in_value_out_idx;
  meta_vec.push_back(OpMeta(OpMeta::Type::Gather, s));
  return true;
}

bool PreLookupOptimizer::MatchSelectValue(NodeDef *&value_node,
                                          int &value_out_idx) {
  RETURN_IF_FALSE(value_node && value_node->op() == "Select");

  NodeDef *in_value_node;
  int in_value_out_idx;
  IntervalSet s;
  RETURN_IF_FALSE(MatchExpr(GetInputNode(value_node, 0), in_value_node,
                            in_value_out_idx, s));

  NodeDef *left = GetInputNode(in_value_node, 1);
  NodeDef *right = GetInputNode(in_value_node, 2);

  if (right == in_value_node) {
    std::swap(left, right);
    s = UniversalSet - s;
  }

  CONVERGE_OR_RETURN(left, in_value_node);

  EXTRACT_SPLAT_CONST_OR_RETURN(right, c, int);

  value_node = in_value_node;
  value_out_idx = in_value_out_idx;
  meta_vec.push_back(OpMeta(OpMeta::Type::Select, s, c));
  return true;
}

bool PreLookupOptimizer::MatchMapValue(NodeDef *&value_node,
                                       int &value_out_idx) {
  RETURN_IF_FALSE(value_node);

  NodeDef *in_value_node;
  int in_value_out_idx;
  if (value_node->op() == "StringToHashBucketFast") {
    int num_buckets = value_node->attr().at("num_buckets").i();
    IntervalSet s(Interval::right_open(0, num_buckets));
    meta_vec.push_back(OpMeta(OpMeta::Type::Map, s));
    in_value_node = GetInputNode(value_node, 0);
    in_value_out_idx = GetOutputIdxByTensor(value_node->input(0));
  } else if (value_node->op() == "Bucketize") {
    int num_buckets = value_node->attr().at("boundaries").list().f_size();
    IntervalSet s(Interval::right_open(0, num_buckets));
    meta_vec.push_back(OpMeta(OpMeta::Type::Map, s));
    in_value_node = GetInputNode(value_node, 0);
    in_value_out_idx = GetOutputIdxByTensor(value_node->input(0));
  } else if (value_node->op() == "StringToNumber") {
    meta_vec.push_back(OpMeta(OpMeta::Type::Map, UniversalSet));
    in_value_node = GetInputNode(value_node, 0);
    in_value_out_idx = GetOutputIdxByTensor(value_node->input(0));
  } else {
    RETURN_FALSE;
  }

  value_reserve_ops.push(value_node);
  value_node = in_value_node;
  value_out_idx = in_value_out_idx;

  return true;
}

bool PreLookupOptimizer::MatchKeepValue(NodeDef *&value_node,
                                        int &value_out_idx) {
  RETURN_IF_FALSE(value_node);

  NodeDef *in_value_node;
  int in_value_out_idx;
  if (value_node->op() == "Reshape" || value_node->op() == "Cast" ||
      value_node->op() == "ExpandDims" || value_node->op() == "Squeeze") {
    meta_vec.push_back(OpMeta(OpMeta::Type::Keep));
    in_value_node = GetInputNode(value_node, 0);
    in_value_out_idx = GetOutputIdxByTensor(value_node->input(0));
  } else {
    RETURN_FALSE;
  }

  value_reserve_ops.push(value_node);
  value_node = in_value_node;
  value_out_idx = in_value_out_idx;

  return true;
}

bool PreLookupOptimizer::MatchSourceValue(NodeDef *&value_node,
                                          int &value_out_idx,
                                          NodeDef *&indice_node,
                                          int &indice_out_idx,
                                          DataType &indice_type) {
  RETURN_IF_FALSE(value_node);

  if (value_node->op() == "Placeholder") {
    if (indice_node)
      indice_type = GetOutputType(indice_node, indice_out_idx);
    RECOM_VLOG << "Match Placeholder source";
  } else if (GetOutputType(value_node, value_out_idx) == DT_STRING) {
    auto match_known_cases = [&]() {
      if (value_node->op() == "StringSplit" ||
          value_node->op() == "StringSplitV2") {
        CONVERGE_OR_RETURN(indice_node, value_node);
        indice_type = DT_INT64;
        RECOM_VLOG << "Match StringSplit source";
      } else if (value_node->op() == "GatherNd") {
        // TODO: split match and reconstruct
        NodeDef *in_value_node = GetInputNode(value_node, 0);
        int in_value_out_idx = GetOutputIdxByTensor(value_node->input(0));
        FIND_INPUT_OR_RETURN(value_node, where, 1, "Where");
        CONVERGE_OR_RETURN(indice_node, where);
        FIND_INPUT_OR_RETURN(where, not_equal, 0, "NotEqual");
        FIND_INPUT_OR_RETURN(not_equal, c, 1, "Const");

        if (in_value_node->op() == "ExpandDims") {
          in_value_out_idx = GetOutputIdxByTensor(in_value_node->input(0));
          in_value_node = GetInputNode(in_value_node, 0);
        }

        Tensor c_t;
        LOG_AND_RETURN_IF_FALSE(
            c_t.FromProto(c->attr().at("value").tensor()),
            "Convert TensorProto of Const op to Tensor fail");
        RETURN_IF_FALSE(c_t.dtype() == DT_STRING);
        RETURN_IF_FALSE(c_t.NumElements() == 1);
        std::string c_val = c_t.flat<tstring>()(0);

        NodeDef *gather_node = gd->add_node();
        gather_node->set_op("Addons>GatherStrValueGenIndice");
        gather_node->set_name(GetNameInnerPrefix(value_node->name()) + "/" +
                              gather_node->op());
        gather_node->add_input(FormTensorName(in_value_node, in_value_out_idx));
        (*gather_node->mutable_attr())["not_equal"].set_s(c_val);
        (*gather_node->mutable_attr())["T"].set_type(DT_INT64);
        RETURN_IF_FALSE(symbolic_context->ShapeKnown(where->name()));
        (*gather_node->mutable_attr())["dense_rank"].set_i(
            symbolic_context->GetShape(where->name()).size());
        indice_node = gather_node;
        indice_out_idx = 0;
        indice_type = DT_INT64;
        value_node = gather_node;
        value_out_idx = 1;
        RECOM_VLOG << "Match GatherStrValueGenIndice source";
      } else {
        RETURN_FALSE;
      }
      return true;
    };
    if (!match_known_cases()) {
      if (indice_node)
        indice_type = GetOutputType(indice_node, indice_out_idx);
      RECOM_VLOG << "Match string value with unknown case, node "
                 << value_node->name();
    }
  } else {
    RETURN_FALSE;
  }

  return true;
}

void PreLookupOptimizer::SkipReserveIndiceOps(NodeDef *&indice_node,
                                              int &indice_out_idx) {
  while (indice_node && (indice_node->op() == "Cast" ||
                         indice_node->op() == "SparseReshape")) {
    indice_reserve_ops.push(indice_node);
    indice_node = GetInputNode(indice_node, 0);
    indice_out_idx = GetOutputIdxByTensor(indice_node->input(0));
  }
}

bool PreLookupOptimizer::MatchBeforeLookup(NodeDef *&value_node,
                                           int &value_out_idx,
                                           NodeDef *&indice_node,
                                           int &indice_out_idx,
                                           DataType &indice_type) {
  SkipReserveIndiceOps(indice_node, indice_out_idx);
  while (!MatchSourceValue(value_node, value_out_idx, indice_node,
                           indice_out_idx, indice_type)) {
    if (value_node && indice_node) {
      RECOM_VLOG << "value_node: [" << value_node->name() << ":"
                 << value_out_idx << ", " << value_node->op()
                 << "], indice_node: [" << indice_node->name() << ":"
                 << indice_out_idx << ", " << indice_node->op() << "]";
    } else if (value_node) {
      RECOM_VLOG << "value_node: [" << value_node->name() << ":"
                 << value_out_idx << ", " << value_node->op() << "]";
    } else {
      RETURN_FALSE;
    }

    if (MatchGatherValue(value_node, value_out_idx, indice_node,
                         indice_out_idx)) {
      RECOM_VLOG << "MatchGatherValue success";
    } else if (MatchSelectValue(value_node, value_out_idx)) {
      RECOM_VLOG << "MatchSelectValue success";
    } else if (MatchMapValue(value_node, value_out_idx)) {
      RECOM_VLOG << "MatchMapValue success";
    } else if (MatchKeepValue(value_node, value_out_idx)) {
      RECOM_VLOG << "MatchKeepValue success";
    } else {
      RETURN_FALSE;
    }
    SkipReserveIndiceOps(indice_node, indice_out_idx);
  }
  return true;
}

bool PreLookupOptimizer::Simplify() {
  std::vector<OpMeta> results;
  IntervalSet global_interval_set = UniversalSet;
  IntervalSet s = UniversalSet;
  for (auto itr = meta_vec.rbegin(); itr != meta_vec.rend(); ++itr) {
    switch (itr->type) {
    case OpMeta::Type::Map: {
      global_interval_set = itr->s;
      results.push_back(*itr);
    } break;
    case OpMeta::Type::Keep: {
      results.push_back(*itr);
    } break;
    case OpMeta::Type::Gather: {
      s = s & itr->s;
      if (itr + 1 == meta_vec.rend() ||
          (itr + 1)->type != OpMeta::Type::Gather) {
        IntervalSet gis = s & global_interval_set;
        if (global_interval_set != gis) {
          int len_s = boost::icl::iterative_size(s);
          int len_gis = boost::icl::iterative_size(gis);
          IntervalSet better_s = len_s <= len_gis ? s : gis;
          results.push_back(OpMeta(OpMeta::Type::Gather, better_s));
          global_interval_set = gis;
        }
        s = UniversalSet;
      }
    } break;
    case OpMeta::Type::Select: {
      s = s & itr->s;
      if (itr + 1 == meta_vec.rend() ||
          (itr + 1)->type != OpMeta::Type::Select || itr->a != (itr + 1)->a) {
        IntervalSet gis = s & global_interval_set;
        if (global_interval_set != gis) {
          int len_s = boost::icl::iterative_size(s);
          int len_gis = boost::icl::iterative_size(gis);
          IntervalSet better_s = len_s <= len_gis ? s : gis;
          results.push_back(OpMeta(OpMeta::Type::Gather, better_s, itr->a));
          global_interval_set = gis | itr->a;
        }
        s = UniversalSet;
      }
    } break;
    }
  }

  for (const OpMeta &meta : results) {
    switch (meta.type) {
    case OpMeta::Type::Map: {
      RECOM_VLOG << "Map " << meta.s;
    } break;
    case OpMeta::Type::Keep: {
      RECOM_VLOG << "Keep";
    } break;
    case OpMeta::Type::Gather: {
      RECOM_VLOG << "Gather " << meta.s;
    } break;
    case OpMeta::Type::Select: {
      RECOM_VLOG << "Select " << meta.a << " " << meta.s;
    } break;
    }
  }

  meta_vec = results;
  return true;
}

bool PreLookupOptimizer::ReconstructGraph(NodeDef *sfer, NodeDef *value_node,
                                          int value_out_idx,
                                          NodeDef *indice_node,
                                          int indice_out_idx,
                                          DataType init_indice_type) {
  RETURN_IF_FALSE(value_node);

  // Cast indice input to DT_INT64
  if (indice_node) {
    if (init_indice_type != DT_INT64) {
      NodeDef *cast = gd->add_node();
      cast->set_op("Cast");
      cast->set_name(indice_node->name() + "_cast");
      (*cast->mutable_attr())["SrcT"].set_type(init_indice_type);
      (*cast->mutable_attr())["DstT"].set_type(DT_INT64);
      cast->add_input(FormTensorName(indice_node, indice_out_idx));

      indice_node = cast;
      indice_out_idx = 0;
    }
  }

  // TODO: currently only support value type be DT_INT64
  // need to record type info in meta and then adjust the attr of
  // reconstructed nodes
  for (const OpMeta &meta : meta_vec) {
    switch (meta.type) {
    case OpMeta::Type::Map:
      RETURN_IF_FALSE(ReconstructMapValue(value_node, value_out_idx, meta));
      break;
    case OpMeta::Type::Keep:
      RETURN_IF_FALSE(ReconstructKeepValue(value_node, value_out_idx, meta));
      break;
    case OpMeta::Type::Gather:
      RETURN_IF_FALSE(ReconstructGatherValue(
          value_node, value_out_idx, indice_node, indice_out_idx, meta));
      break;
    case OpMeta::Type::Select:
      RETURN_IF_FALSE(ReconstructSelectValue(value_node, value_out_idx, meta));
      break;
    }
  }

  RETURN_IF_FALSE(ReconstructIndice(indice_node, indice_out_idx));
  RETURN_IF_FALSE(ReconnectToSFER(sfer, value_node, value_out_idx, indice_node,
                                  indice_out_idx));

  return true;
}

bool PreLookupOptimizer::ReconstructKeepValue(NodeDef *&value_node,
                                              int &value_out_idx,
                                              const OpMeta &meta) {
  RETURN_IF_FALSE(meta.type == OpMeta::Type::Keep);
  RETURN_IF_FALSE(value_node);

  NodeDef *node = value_reserve_ops.top();
  value_reserve_ops.pop();

  if (node->op() == "Reshape" || node->op() == "Cast" ||
      node->op() == "ExpandDims" || node->op() == "Squeeze") {
    NodeDef *copy_node = gd->add_node();
    copy_node->CopyFrom(*node);
    copy_node->set_name(node->name() + "_new");
    *(copy_node->mutable_input(0)) = FormTensorName(value_node, value_out_idx);
    value_node = copy_node;
    value_out_idx = 0;
  } else {
    RETURN_FALSE;
  }

  return true;
}

bool PreLookupOptimizer::ReconstructMapValue(NodeDef *&value_node,
                                             int &value_out_idx,
                                             const OpMeta &meta) {
  RETURN_IF_FALSE(meta.type == OpMeta::Type::Map);
  RETURN_IF_FALSE(value_node);

  NodeDef *node = value_reserve_ops.top();
  value_reserve_ops.pop();

  if (node->op() == "StringToHashBucketFast" || node->op() == "Bucketize" ||
      node->op() == "StringToNumber") {
    NodeDef *copy_node = gd->add_node();
    copy_node->CopyFrom(*node);
    copy_node->set_name(node->name() + "_new");
    *(copy_node->mutable_input(0)) = FormTensorName(value_node, value_out_idx);
    value_node = copy_node;
    value_out_idx = 0;
  } else {
    RETURN_FALSE;
  }

  return true;
}

std::vector<std::pair<int, int>>
PreLookupOptimizer::GetClosedBoundaries(const IntervalSet &s) {
  std::vector<std::pair<int, int>> close_boundaries;
  for (const auto &i : s) {
    int l = i.lower();
    int u = i.upper();
    if (i.bounds() == IntervalBounds::open()) {
      ++l, --u;
    } else if (i.bounds() == IntervalBounds::left_open()) {
      ++l;
    } else if (i.bounds() == IntervalBounds::right_open()) {
      --u;
    }
    close_boundaries.push_back(std::make_pair(l, u));
  }
  return close_boundaries;
}

bool PreLookupOptimizer::ReconstructGatherValue(NodeDef *&value_node,
                                                int &value_out_idx,
                                                NodeDef *&indice_node,
                                                int &indice_out_idx,
                                                const OpMeta &meta) {
  RETURN_IF_FALSE(meta.type == OpMeta::Type::Gather);
  RETURN_IF_FALSE(value_node);

  const auto close_boundaries = GetClosedBoundaries(meta.s);
  NodeDef *gather_node = gd->add_node();
  gather_node->set_op(indice_node ? "Addons>GatherIndiceValue"
                                  : "Addons>GatherValueGenIndice");
  gather_node->set_name(GetNameInnerPrefix(value_node->name()) + "/" +
                        gather_node->op());
  if (indice_node)
    gather_node->add_input(FormTensorName(indice_node, indice_out_idx));
  gather_node->add_input(FormTensorName(value_node, value_out_idx));
  auto &left_boundaries_attr =
      (*gather_node->mutable_attr())["left_boundaries"];
  auto &right_boundaries_attr =
      (*gather_node->mutable_attr())["right_boundaries"];
  for (const auto &boundary : close_boundaries) {
    left_boundaries_attr.mutable_list()->add_i(boundary.first);
    right_boundaries_attr.mutable_list()->add_i(boundary.second);
  }
  indice_node = gather_node;
  indice_out_idx = 0;
  value_node = gather_node;
  value_out_idx = 1;

  return true;
}

bool PreLookupOptimizer::ReconstructSelectValue(NodeDef *&value_node,
                                                int &value_out_idx,
                                                const OpMeta &meta) {
  RETURN_IF_FALSE(meta.type == OpMeta::Type::Select);
  RETURN_IF_FALSE(value_node);

  const auto close_boundaries = GetClosedBoundaries(meta.s);
  NodeDef *select_node = gd->add_node();
  select_node->set_op("Addons>SelectValue");
  select_node->set_name(GetNameInnerPrefix(value_node->name()) + "/" +
                        select_node->op());
  select_node->add_input(FormTensorName(value_node, value_out_idx));
  auto &left_boundaries_attr =
      (*select_node->mutable_attr())["left_boundaries"];
  auto &right_boundaries_attr =
      (*select_node->mutable_attr())["right_boundaries"];
  for (const auto &boundary : close_boundaries) {
    left_boundaries_attr.mutable_list()->add_i(boundary.first);
    right_boundaries_attr.mutable_list()->add_i(boundary.second);
  }
  (*select_node->mutable_attr())["substitute"].set_i(meta.a);
  value_node = select_node;
  value_out_idx = 0;

  return true;
}

bool PreLookupOptimizer::ReconstructIndice(NodeDef *&indice_node,
                                           int &indice_out_idx) {
  RETURN_IF_FALSE(indice_node);

  HashMapT<std::string, std::vector<NodeDef *>> op_nodes_mapping;
  while (!indice_reserve_ops.empty()) {
    NodeDef *node = indice_reserve_ops.top();
    RETURN_IF_FALSE(node->op() == "Cast" || node->op() == "SparseReshape");
    op_nodes_mapping[node->op()].push_back(node);
    indice_reserve_ops.pop();
  }

  if (op_nodes_mapping.count("SparseReshape")) {
    std::vector<NodeDef *> nodes = op_nodes_mapping.at("SparseReshape");
    RETURN_IF_FALSE(symbolic_context->ContentKnown(nodes[0]->input(1)));
    RETURN_IF_FALSE(symbolic_context->ContentKnown(nodes.back()->input(2)));
    const ExprVec &orig_shape =
        symbolic_context->GetContent(nodes[0]->input(1));
    const ExprVec &new_shape =
        symbolic_context->GetContent(nodes.back()->input(2));

    if (orig_shape != new_shape) {
      NodeDef *orig_shape_node = ConstructShapeNodeByExpr(
          symbolic_context->GetContent(nodes[0]->input(1)),
          GetInputType(nodes.back(), 1),
          nodes[0]->name() + "/orig_shape_construct");

      NodeDef *new_shape_node = ConstructShapeNodeByExpr(
          symbolic_context->GetContent(nodes.back()->input(2)),
          GetInputType(nodes.back(), 2),
          nodes.back()->name() + "/new_shape_construct");

      NodeDef *sparse_reshape_node = gd->add_node();
      sparse_reshape_node->set_op("SparseReshape");
      sparse_reshape_node->set_name(indice_node->name() +
                                    "/Reconstructed_SparseReshape");
      sparse_reshape_node->add_input(
          FormTensorName(indice_node, indice_out_idx));
      sparse_reshape_node->add_input(orig_shape_node->name());
      sparse_reshape_node->add_input(new_shape_node->name());

      indice_node = sparse_reshape_node;
      indice_out_idx = 0;
    }
  }

  return true;
}

bool PreLookupOptimizer::ReconnectToSFER(NodeDef *sfer, NodeDef *&value_node,
                                         int &value_out_idx,
                                         NodeDef *&indice_node,
                                         int &indice_out_idx) {
  RETURN_IF_FALSE(value_node && indice_node);

  *(sfer->mutable_input(0)) = FormTensorName(indice_node, indice_out_idx);
  *(sfer->mutable_input(1)) = FormTensorName(value_node, value_out_idx);

  NodeDef *shape_node = ConstructShapeNodeByExpr(
      symbolic_context->GetContent(sfer->input(2)), GetInputType(sfer, 2),
      sfer->name() + "/shape_construct");
  *(sfer->mutable_input(2)) = shape_node->name();

  return true;
}

} // namespace feature_opt
} // namespace tensorflow