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

#include "lookup_optimizer.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

void LookupOptimizer::Optimize() {
  for (int i = 0; i < fc_node_sets.size(); ++i) {
    for (const std::string &node_name : fc_node_sets[i]) {
      NodeDef *node = node_mapping.at(node_name);
      if (node->op() == "SparseSegmentMean" ||
          node->op() == "SparseSegmentSum") {
        DominantNodes dominant;
        if (Match(node, dominant)) {
          RECOM_VLOG << "Match Lookup successfully";
          if (MatchDenseInput(dominant)) {
            RECOM_VLOG << "Match Dense Input successfully";
            if (RewriteDenseInput(dominant, i)) {
              RECOM_VLOG << "Rewrite Dense Input successfully";
            } else {
              RECOM_VLOG << "Fail to rewrite Dense Input";
            }
          } else if (MatchGatherScatter(dominant)) {
            RECOM_VLOG << "Match Gather Scatter successfully";
            if (RewriteGatherScatter(dominant, i)) {
              RECOM_VLOG << "Rewrite Gather Scatter successfully";
            } else {
              RECOM_VLOG << "Fail to rewrite Gather Scatter";
            }
          } else {
            if (RewriteSeedWithNumSegments(dominant, i)) {
              RECOM_VLOG << "Add SeedWithNumSegments successfully";
            } else {
              RECOM_VLOG << "Fail to add SeedWithNumSegments";
            }
          }
        }
        break;
      }
    }
  }
}

bool LookupOptimizer::Match(NodeDef *sparse_segment_node,
                            DominantNodes &dominant) {
  FIND_INPUT_OR_RETURN(sparse_segment_node, suspicious_weight, 0,
                       "ResourceGather,GatherV2,VariableV2,Const,VarHandleOp");
  dominant.seed = sparse_segment_node;

  // TODO: check strided_slice function
  NodeDef *segment_ids_input = GetInputNode(sparse_segment_node, 2);
  NodeDef *strided_slice = segment_ids_input->op() == "Cast"
                               ? GetInputNode(segment_ids_input, 0)
                               : segment_ids_input;
  RETURN_IF_FALSE(strided_slice->op() == "StridedSlice");
  RETURN_IF_FALSE(strided_slice->attr().at("begin_mask").i() == 1);
  RETURN_IF_FALSE(strided_slice->attr().at("ellipsis_mask").i() == 0);
  RETURN_IF_FALSE(strided_slice->attr().at("end_mask").i() == 1);
  RETURN_IF_FALSE(strided_slice->attr().at("new_axis_mask").i() == 0);
  RETURN_IF_FALSE(strided_slice->attr().at("shrink_axis_mask").i() == 2);
  EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(strided_slice, 1), ss_begin, int);
  RETURN_IF_FALSE(ss_begin.size() == 2 && ss_begin[0] == 0);
  EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(strided_slice, 3), ss_stride,
                                 int);
  RETURN_IF_FALSE(ss_stride.size() == 2 && ss_stride[1] == 1);
  dominant.slice = strided_slice;

  FIND_INPUT_OR_RETURN(strided_slice, sfer_tmp, 0, "SparseFillEmptyRows");
  dominant.sfer = sfer_tmp;

  if (suspicious_weight->op() == "ResourceGather" ||
      suspicious_weight->op() == "GatherV2") {
    FIND_INPUT_OR_RETURN(sparse_segment_node, unique_node, 1, "Unique");
    FIND_INPUT_OR_RETURN(suspicious_weight, unique_node_x1, 1, "Unique");
    CONVERGE_OR_RETURN(unique_node, unique_node_x1);

    FIND_INPUT_OR_RETURN(unique_node, sfer_x1, 0, "SparseFillEmptyRows");
    FIND_INPUT_OR_RETURN(suspicious_weight, actual_weight, 0,
                         "VariableV2,Const,VarHandleOp");
    CONVERGE_OR_RETURN(dominant.sfer, sfer_x1);
    dominant.weight = actual_weight;
  } else {
    FIND_INPUT_OR_RETURN(sparse_segment_node, sfer_x1, 1,
                         "SparseFillEmptyRows");
    CONVERGE_OR_RETURN(dominant.sfer, sfer_x1);
    dominant.weight = suspicious_weight;
  }

  // SparseSegmentAggregate output graph
  FIND_OUTPUT_OR_RETURN(sparse_segment_node, final_select, "Select,SelectV2");
  dominant.select = final_select;

  FIND_INPUT_OR_RETURN(final_select, candidate_x1, 2,
                       sparse_segment_node->op());
  CONVERGE_OR_RETURN(sparse_segment_node, candidate_x1);

  FIND_INPUT_OR_RETURN(final_select, zeros_like, 1, "ZerosLike");
  FIND_INPUT_OR_RETURN(zeros_like, candidate_x2, 0, sparse_segment_node->op());
  CONVERGE_OR_RETURN(sparse_segment_node, candidate_x2);

  FIND_INPUT_OR_RETURN(final_select, tile, 0, "Tile");
  FIND_INPUT_OR_RETURN(tile, reshape, 0, "Reshape");
  FIND_INPUT_OR_RETURN(reshape, sfer_x1, 0, "SparseFillEmptyRows");
  CONVERGE_OR_RETURN(dominant.sfer, sfer_x1);

  // We do not need to check the multiples input of Tile because the Select Op
  // can guaranntee it after checking the shape of the condition input of Tile
  RETURN_IF_FALSE(symbolic_context->ShapeKnown(reshape->name()));
  RETURN_IF_FALSE(symbolic_context->ShapeKnown(reshape->input(0)));
  const ExprVec reshape_shape = symbolic_context->GetShape(reshape->name());
  const ExprVec before_reshape = symbolic_context->GetShape(reshape->input(0));
  RETURN_IF_FALSE(before_reshape.size() == 1 && reshape_shape.size() == 2);
  RETURN_IF_FALSE(before_reshape[0] == reshape_shape[0] &&
                  reshape_shape[1] == 1);

  return true;
}

bool LookupOptimizer::MatchDenseInput(DominantNodes dominant) {
  RETURN_IF_FALSE(symbolic_context->ShapeKnown(dominant.sfer->input(1)));
  RETURN_IF_FALSE(symbolic_context->ContentKnown(dominant.sfer->input(2)));
  const ExprVec sp_value_shape =
      symbolic_context->GetShape(dominant.sfer->input(1));
  const ExprVec dense_shape =
      symbolic_context->GetContent(dominant.sfer->input(2));
  const Expression num_dense_elements =
      std::accumulate(dense_shape.begin(), dense_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  return num_dense_elements == sp_value_shape[0];
}

bool LookupOptimizer::MatchGatherScatter(DominantNodes dominant) {
  RETURN_IF_FALSE(symbolic_context->ContentKnown(dominant.sfer->input(2)));
  const ExprVec dense_shape =
      symbolic_context->GetContent(dominant.sfer->input(2));
  return dense_shape.back() == Expression(1);
}

bool LookupOptimizer::RewriteSeedWithNumSegments(DominantNodes dominant,
                                                 int fc_id) {
  RETURN_IF_FALSE(symbolic_context->ContentKnown(dominant.sfer->input(2)));
  const ExprVec &dense_shape =
      symbolic_context->GetContent(dominant.sfer->input(2));

  NodeDef *seed_with_num_segments = gd->add_node();
  seed_with_num_segments->set_name(dominant.seed->name() +
                                   "_with_num_segments");
  seed_with_num_segments->set_op(dominant.seed->op() + "WithNumSegments");

  auto &attr = *seed_with_num_segments->mutable_attr();
  attr["T"].set_type(GetOutputType(dominant.seed, 0));
  attr["Tidx"].set_type(GetInputType(dominant.sfer, 1));
  attr["Tsegmentids"].set_type(GetInputType(dominant.sfer, 0));
  attr["Tnumsegments"].set_type(DT_INT64);

  seed_with_num_segments->add_input(dominant.weight->name());
  seed_with_num_segments->add_input(dominant.sfer->input(1));

  // slice the indices to get the segment_ids
  NodeDef *indices_node = GetInputNode(dominant.sfer, 0);
  if (indices_node->op() == "Addons>GatherStrValueGenIndice") {
    NodeDef *new_indices_node = gd->add_node();
    new_indices_node->CopyFrom(*indices_node);
    new_indices_node->set_name(indices_node->name() + "_new");
    (*new_indices_node->mutable_attr())["dense_rank"].set_i(1);

    NodeDef *squeeze = gd->add_node();
    squeeze->set_name(new_indices_node->name() + "_squeeze");
    squeeze->set_op("Squeeze");
    (*squeeze->mutable_attr())["T"].set_type(GetOutputType(indices_node, 0));
    (*squeeze->mutable_attr())["squeeze_dims"].mutable_list()->add_i(-1);
    squeeze->add_input(new_indices_node->name());

    seed_with_num_segments->add_input(squeeze->name());

    for (const std::string &output_name :
         out_mapping.at(indices_node->name())) {
      NodeDef *output_node = node_mapping.at(output_name);
      for (int i = 0; i < output_node->input_size(); ++i) {
        if (GetNodeNameByTensor(output_node->input(i)) ==
            indices_node->name()) {
          int idx = GetOutputIdxByTensor(output_node->input(i));
          if (idx != 0) {
            *(output_node->mutable_input(i)) =
                FormTensorName(new_indices_node, idx);
          }
        }
      }
    }
  } else {
#define ADD_PAIR_CONST(node, val1, val2)                                       \
  NodeDef *node = gd->add_node();                                              \
  node->set_op("Const");                                                       \
  node->set_name(GetNameInnerPrefix(dominant.sfer->input(0)) +                 \
                 "/added_strided_slice/" + #node);                             \
  AttrValue node##_value;                                                      \
  node##_value.mutable_tensor()->set_dtype(DT_INT64);                          \
  node##_value.mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(  \
      2);                                                                      \
  node##_value.mutable_tensor()->add_int64_val(val1);                          \
  node##_value.mutable_tensor()->add_int64_val(val2);                          \
  (*node->mutable_attr())["value"] = node##_value;                             \
  (*node->mutable_attr())["dtype"].set_type(DT_INT64);

    ADD_PAIR_CONST(begin_node, 0, 0);
    ADD_PAIR_CONST(end_node, 0, 1);
    ADD_PAIR_CONST(stride_node, 1, 1);

#undef ADD_PAIR_CONST

    NodeDef *strided_slice = gd->add_node();
    strided_slice->set_op("StridedSlice");
    strided_slice->set_name(GetNameInnerPrefix(dominant.sfer->input(0)) +
                            "/added_strided_slice");
    strided_slice->add_input(dominant.sfer->input(0));
    strided_slice->add_input(begin_node->name());
    strided_slice->add_input(end_node->name());
    strided_slice->add_input(stride_node->name());
    (*strided_slice->mutable_attr())["T"].set_type(
        GetInputType(dominant.sfer, 0));
    (*strided_slice->mutable_attr())["Index"].set_type(DT_INT64);
    (*strided_slice->mutable_attr())["begin_mask"].set_i(1);
    (*strided_slice->mutable_attr())["end_mask"].set_i(1);
    (*strided_slice->mutable_attr())["shrink_axis_mask"].set_i(2);
    seed_with_num_segments->add_input(strided_slice->name());
  }

  NodeDef *num_segments_node = ConstructShapeNodeByExpr(
      {dense_shape[0]}, DT_INT64, dominant.seed->name() + "/num_segments");
  NodeDef *squeeze = gd->add_node();
  squeeze->set_name(num_segments_node->name() + "_squeeze");
  squeeze->set_op("Squeeze");
  (*squeeze->mutable_attr())["T"].set_type(DT_INT64);
  (*squeeze->mutable_attr())["squeeze_dims"].mutable_list()->add_i(0);
  squeeze->add_input(num_segments_node->name());
  seed_with_num_segments->add_input(squeeze->name());

  for (const std::string &output_name :
       out_mapping.at(dominant.select->name())) {
    NodeDef *output_node = node_mapping.at(output_name);
    for (int i = 0; i < output_node->input_size(); ++i) {
      if (GetNodeNameByTensor(output_node->input(i)) ==
          dominant.select->name()) {
        *(output_node->mutable_input(i)) = seed_with_num_segments->name();
      }
    }
  }

  return true;
}

bool LookupOptimizer::RewriteDenseInput(DominantNodes dominant, int fc_id) {
  NodeDef *zero_scalar = gd->add_node();
  zero_scalar->set_op("Const");
  zero_scalar->set_name(GetNameInnerPrefix(dominant.seed->name()) +
                        "/GatherDense/zero");
  (*zero_scalar->mutable_attr())["value"].mutable_tensor()->set_dtype(DT_INT32);
  (*zero_scalar->mutable_attr())["value"].mutable_tensor()->add_int_val(0);
  (*zero_scalar->mutable_attr())["dtype"].set_type(DT_INT32);

  NodeDef *gather_dense = gd->add_node();
  gather_dense->set_op("GatherV2");
  gather_dense->set_name(GetNameInnerPrefix(dominant.seed->name()) +
                         "/GatherDense");
  gather_dense->add_input(dominant.weight->name());
  gather_dense->add_input(dominant.sfer->input(1));
  gather_dense->add_input(zero_scalar->name());
  (*gather_dense->mutable_attr())["Tparams"].set_type(
      GetOutputType(dominant.seed, 0));
  (*gather_dense->mutable_attr())["Tindices"].set_type(
      dominant.sfer->attr().at("T").type());
  (*gather_dense->mutable_attr())["Taxis"].set_type(
      zero_scalar->attr().at("dtype").type());

  RETURN_IF_FALSE(symbolic_context->ContentKnown(dominant.sfer->input(2)));
  const ExprVec dense_shape =
      symbolic_context->GetContent(dominant.sfer->input(2));

  NodeDef *new_output = gather_dense;
  if (dense_shape.back() != Expression(1)) {
    NodeDef *reduce_node = gd->add_node();
    if (dominant.seed->op() == "SparseSegmentMean") {
      reduce_node->set_op("Mean");
    } else if (dominant.seed->op() == "SparseSegmentSum") {
      reduce_node->set_op("Sum");
    }
    LOG_AND_RETURN_FALSE(
        "TODO: support Dense Input Lookup with column length != 1");
    new_output = reduce_node;
  }

  for (const std::string &output_name :
       out_mapping.at(dominant.select->name())) {
    NodeDef *output_node = node_mapping.at(output_name);
    for (int i = 0; i < output_node->input_size(); ++i) {
      if (GetNodeNameByTensor(output_node->input(i)) ==
          dominant.select->name()) {
        *(output_node->mutable_input(i)) = new_output->name();
      }
    }
  }

  return true;
}

bool LookupOptimizer::RewriteGatherScatter(DominantNodes dominant, int fc_id) {
  NodeDef *zero_scalar = gd->add_node();
  zero_scalar->set_op("Const");
  zero_scalar->set_name(GetNameInnerPrefix(dominant.seed->name()) +
                        "/GatherDense/zero");
  (*zero_scalar->mutable_attr())["value"].mutable_tensor()->set_dtype(DT_INT32);
  (*zero_scalar->mutable_attr())["value"].mutable_tensor()->add_int_val(0);
  (*zero_scalar->mutable_attr())["dtype"].set_type(DT_INT32);

  NodeDef *gather = gd->add_node();
  gather->set_op("GatherV2");
  gather->set_name(GetNameInnerPrefix(dominant.seed->name()) +
                   "/GatherScatter/Gather");
  gather->add_input(dominant.weight->name());
  gather->add_input(dominant.sfer->input(1));
  gather->add_input(zero_scalar->name());
  (*gather->mutable_attr())["Tparams"].set_type(
      GetOutputType(dominant.seed, 0));
  (*gather->mutable_attr())["Tindices"].set_type(
      dominant.sfer->attr().at("T").type());
  (*gather->mutable_attr())["Taxis"].set_type(
      zero_scalar->attr().at("dtype").type());

  NodeDef *scatter = gd->add_node();
  scatter->set_op("ScatterNd");
  scatter->set_name(GetNameInnerPrefix(dominant.seed->name()) +
                    "/GatherScatter/Scatter");
  (*scatter->mutable_attr())["T"].set_type(GetOutputType(dominant.seed, 0));
  (*scatter->mutable_attr())["Tindices"].set_type(
      GetInputType(dominant.sfer, 0));

  // slice the indices to get the segment_ids
  NodeDef *indices_node = GetInputNode(dominant.sfer, 0);
  if (indices_node->op() == "Addons>GatherStrValueGenIndice") {
    NodeDef *new_indices_node = gd->add_node();
    new_indices_node->CopyFrom(*indices_node);
    new_indices_node->set_name(indices_node->name() + "_new");
    (*new_indices_node->mutable_attr())["dense_rank"].set_i(1);
    scatter->add_input(new_indices_node->name());

    for (const std::string &output_name :
         out_mapping.at(indices_node->name())) {
      NodeDef *output_node = node_mapping.at(output_name);
      for (int i = 0; i < output_node->input_size(); ++i) {
        if (GetNodeNameByTensor(output_node->input(i)) ==
            indices_node->name()) {
          int idx = GetOutputIdxByTensor(output_node->input(i));
          if (idx != 0) {
            *(output_node->mutable_input(i)) =
                FormTensorName(new_indices_node, idx);
          }
        }
      }
    }
  } else {
#define ADD_PAIR_CONST(node, val1, val2)                                       \
  NodeDef *node = gd->add_node();                                              \
  node->set_op("Const");                                                       \
  node->set_name(GetNameInnerPrefix(dominant.sfer->input(0)) +                 \
                 "/added_strided_slice/" + #node);                             \
  AttrValue node##_value;                                                      \
  node##_value.mutable_tensor()->set_dtype(DT_INT64);                          \
  node##_value.mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(  \
      2);                                                                      \
  node##_value.mutable_tensor()->add_int64_val(val1);                          \
  node##_value.mutable_tensor()->add_int64_val(val2);                          \
  (*node->mutable_attr())["value"] = node##_value;                             \
  (*node->mutable_attr())["dtype"].set_type(DT_INT64);

    ADD_PAIR_CONST(begin_node, 0, 0);
    ADD_PAIR_CONST(end_node, 0, 1);
    ADD_PAIR_CONST(stride_node, 1, 1);

#undef ADD_PAIR_CONST

    NodeDef *strided_slice = gd->add_node();
    strided_slice->set_op("StridedSlice");
    strided_slice->set_name(GetNameInnerPrefix(dominant.sfer->input(0)) +
                            "/added_strided_slice");
    strided_slice->add_input(dominant.sfer->input(0));
    strided_slice->add_input(begin_node->name());
    strided_slice->add_input(end_node->name());
    strided_slice->add_input(stride_node->name());
    (*strided_slice->mutable_attr())["T"].set_type(
        GetInputType(dominant.sfer, 0));
    (*strided_slice->mutable_attr())["Index"].set_type(DT_INT64);
    (*strided_slice->mutable_attr())["begin_mask"].set_i(1);
    (*strided_slice->mutable_attr())["end_mask"].set_i(1);
    scatter->add_input(strided_slice->name());
  }

  scatter->add_input(gather->name());

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(dominant.seed->input(0)));
  RETURN_IF_FALSE(symbolic_context->ContentKnown(dominant.sfer->input(2)));
  const ExprVec &dense_shape =
      symbolic_context->GetContent(dominant.sfer->input(2));
  const ExprVec &weight_shape =
      symbolic_context->GetShape(dominant.seed->input(0));
  const ExprVec output_shape = {dense_shape[0], weight_shape[1]};
  NodeDef *scatter_shape_node = ConstructShapeNodeByExpr(
      output_shape, GetInputType(dominant.sfer, 0), scatter->name() + "_shape");
  scatter->add_input(scatter_shape_node->name());

  for (const std::string &output_name :
       out_mapping.at(dominant.select->name())) {
    NodeDef *output_node = node_mapping.at(output_name);
    for (int i = 0; i < output_node->input_size(); ++i) {
      if (GetNodeNameByTensor(output_node->input(i)) ==
          dominant.select->name()) {
        *(output_node->mutable_input(i)) = scatter->name();
      }
    }
  }

  return true;
}

// deprecated
bool LookupOptimizer::RewriteExtendedSparse(DominantNodes dominant, int fc_id) {
  const int embedd_dim = FetchGrapplerOutputShapes(dominant.weight)[0].back();

#define ADD_CONST(node, val)                                                   \
  NodeDef *node = gd->add_node();                                              \
  node->set_op("Const");                                                       \
  node->set_name(GetNameInnerPrefix(dominant.seed->name()) +                   \
                 "/added_strided_slice/" + #node);                             \
  AttrValue node##_value;                                                      \
  node##_value.mutable_tensor()->set_dtype(DT_INT64);                          \
  node##_value.mutable_tensor()->mutable_tensor_shape()->add_dim()->set_size(  \
      1);                                                                      \
  node##_value.mutable_tensor()->add_int64_val(val);                           \
  (*node->mutable_attr())["value"] = node##_value;                             \
  (*node->mutable_attr())["dtype"].set_type(DT_INT64);

  ADD_CONST(zero_node, 0);
  ADD_CONST(neg_one_node, -1);
  ADD_CONST(one_node, 1);
  ADD_CONST(embedd_dim_node, embedd_dim);

#undef ADD_CONST

  NodeDef *exssr = gd->add_node();
  exssr->set_op("Addons>Extended" + dominant.seed->op());
  exssr->set_name(GetNameInnerPrefix(dominant.seed->name()) + "/" +
                  exssr->op());

  auto &attr = *exssr->mutable_attr();
  attr["T"].set_type(dominant.weight->attr().at("dtype").type());
  attr["Tindices"].set_type(DT_INT64);
  attr["Tspvalues"].set_type(dominant.sfer->attr().at("T").type());
  attr["Tshape"].set_type(DT_INT64);

  exssr->add_input(dominant.weight->name());
  exssr->add_input(dominant.sfer->input(0));
  exssr->add_input(dominant.sfer->input(1));
  exssr->add_input(dominant.sfer->input(2));

  // slice the shape
  NodeDef *strided_slice = gd->add_node();
  strided_slice->set_op("StridedSlice");
  strided_slice->set_name(GetNameInnerPrefix(dominant.seed->name()) +
                          "/added_strided_slice");
  strided_slice->add_input(dominant.sfer->input(2));
  strided_slice->add_input(zero_node->name());
  strided_slice->add_input(neg_one_node->name());
  strided_slice->add_input(one_node->name());
  (*strided_slice->mutable_attr())["T"].set_type(DT_INT64);
  (*strided_slice->mutable_attr())["Index"].set_type(DT_INT64);

  exssr->add_input(strided_slice->name());

  NodeDef *to_dense = gd->add_node();
  to_dense->set_op("Addons>ExtendedSparseToDense");
  to_dense->set_name(GetNameInnerPrefix(dominant.seed->name()) + "/" +
                     to_dense->op());
  to_dense->add_input(FormTensorName(exssr, 0));
  to_dense->add_input(FormTensorName(exssr, 1));
  to_dense->add_input(exssr->input(4));
  (*to_dense->mutable_attr())["T"].set_type(exssr->attr().at("T").type());
  (*to_dense->mutable_attr())["Tindices"].set_type(
      exssr->attr().at("Tindices").type());
  (*to_dense->mutable_attr())["Tshape"].set_type(
      exssr->attr().at("Tshape").type());
  (*to_dense->mutable_attr())["default_float"].set_f(0);

  for (const std::string &output_name :
       out_mapping.at(dominant.select->name())) {
    NodeDef *output_node = node_mapping.at(output_name);
    for (int i = 0; i < output_node->input_size(); ++i) {
      if (GetNodeNameByTensor(output_node->input(i)) ==
          dominant.select->name()) {
        *(output_node->mutable_input(i)) = to_dense->name();
      }
    }
  }

  fc_node_sets[fc_id].insert(exssr->name());
  fc_node_sets[fc_id].insert(to_dense->name());
  node_mapping[exssr->name()] = exssr;
  node_mapping[to_dense->name()] = to_dense;

  RECOM_VLOG << "Add " << exssr->name() << " " << exssr->op() << " to FC "
             << fc_id;
  RECOM_VLOG << "Add " << to_dense->name() << " " << to_dense->op() << " to FC "
             << fc_id;

  RETURN_IF_FALSE(symbolic_context->ShapeKnown(dominant.sfer->input(1)));
  RETURN_IF_FALSE(symbolic_context->ShapeKnown(dominant.select->name()));
  const Expression nnz = symbolic_context->GetShape(dominant.sfer->input(1))[0];
  const ExprVec dense_shape =
      symbolic_context->GetShape(dominant.select->name());
  const ExprVec dense_prefix(dense_shape.begin(), dense_shape.end() - 1);
  symbolic_context->SetShape(to_dense->name(), dense_shape);
  symbolic_context->SetShape(to_dense->input(0),
                             {nnz, Expression(dense_prefix.size())});
  symbolic_context->SetShape(to_dense->input(1), {nnz, Expression(embedd_dim)});
  symbolic_context->SetShape(to_dense->input(2),
                             {Expression(dense_prefix.size())});
  symbolic_context->SetContent(to_dense->input(2), dense_prefix);
  out_mapping[exssr->name()].insert(to_dense->name());
  out_mapping[to_dense->name()] = out_mapping[dominant.select->name()];

  return true;
}

} // namespace feature_opt
} // namespace tensorflow