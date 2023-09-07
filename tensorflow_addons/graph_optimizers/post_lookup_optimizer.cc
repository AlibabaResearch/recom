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

#include <cassert>
#include <cstddef>
#include <functional>
#include <numeric>
#include <string>
#include <tensorflow/core/framework/node_def.pb.h>
#include <tensorflow/core/framework/types.pb.h>
#include <vector>

#include "fc_optimizer_base.h"
#include "post_lookup_optimizer.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

bool PostLookupOptimizer::GetExpectDensePrefixAndValueShape(
    const ExSpInfo &info, ExprVec &expect_prefix, ExprVec &expect_value_shape) {
  Expression left = info.element_size;
  int pos;
  for (pos = info.dense_shape.size() - 1; pos >= 0; --pos) {
    left /= info.dense_shape[pos];
    if (left == Expression(1))
      break;
  }
  RETURN_IF_FALSE(pos >= 0);

  expect_prefix = ExprVec(pos);
  std::copy(info.dense_shape.cbegin(), info.dense_shape.cbegin() + pos,
            expect_prefix.begin());

  expect_value_shape = ExprVec(info.dense_shape.size() - pos + 1);
  expect_value_shape[0] = info.value_shape[0];
  std::copy(info.dense_shape.cbegin() + pos, info.dense_shape.cend(),
            expect_value_shape.begin() + 1);

  return true;
}

void PostLookupOptimizer::InitPostGraphContext(NodeDef *to_dense_node, int i,
                                               PostGraphContext &context) {
  std::stack<NodeDef *> node_stack;
  node_stack.push(to_dense_node);
  while (!node_stack.empty()) {
    NodeDef *node = node_stack.top();
    node_stack.pop();

    for (const std::string &oname : out_mapping.at(node->name())) {
      if (fc_node_sets[i].count(oname)) {
        NodeDef *onode = node_mapping.at(oname);
        const std::string op = onode->op();
        if (op == "Shape" || op == "ZerosLike" || op == "OnesLike") {
          context.node_cnt_mapping[oname] = 1;
        } else {
          if (context.node_cnt_mapping[oname]++ == 0) {
            node_stack.push(onode);
          }
        }
      } else {
        RECOM_VLOG << "FC " << i << " do not contain " << oname;
      }
    }
  }

  context.to_dense_cpy_cnt = 0;
}

void PostLookupOptimizer::Optimize() {
  for (int i = 0; i < fc_node_sets.size(); ++i) {
    RECOM_VLOG << "Process FC " << i;
    for (const std::string &name : fc_node_sets[i]) {
      NodeDef *node = node_mapping.at(name);
      if (node->op() == "Addons>ExtendedSparseToDense") {
        RECOM_VLOG << "Found " << node->name();

        PostGraphContext context;
        InitPostGraphContext(node, i, context);

        const ExprVec value_shape = symbolic_context->GetShape(node->input(1));
        ExSpInfo curr = {symbolic_context->GetContent(node->input(2)),
                         value_shape, symbolic_context->GetShape(node->name()),
                         std::accumulate(value_shape.begin() + 1,
                                         value_shape.end(), Expression(1),
                                         std::multiplies<Expression>()),
                         node->attr().at("default_float").f()};
        if (Optimize(node, node, 0, curr, context)) {
          RECOM_VLOG << "Optimize " << node->name() << " sucessfully";
        } else {
          RECOM_VLOG << "Fail to optimize " << node->name();
        }
        break;
      }
    }
  }
}

bool PostLookupOptimizer::Optimize(NodeDef *to_dense_node, NodeDef *onode,
                                   int onode_inid, ExSpInfo curr,
                                   PostGraphContext &context) {
  if (onode != to_dense_node) {
    if (++context.curr_cnt_mapping[onode->name()] <
        context.node_cnt_mapping[onode->name()]) {
      RECOM_VLOG << "Waiting " << onode->name() << ", need "
                 << context.node_cnt_mapping[onode->name()] << ", current "
                 << context.curr_cnt_mapping[onode->name()];
      context.pending_nodes[onode->name()][onode_inid] = {to_dense_node, curr};
      return true;
    }

    if (to_dense_node) {
      if (MatchAndRecordReshape(to_dense_node, onode, onode_inid, curr,
                                context)) {
        RECOM_VLOG << "MatchAndRecordReshape successfully";
      } else if (MatchAndRewriteMatMul(to_dense_node, onode, onode_inid, curr,
                                       context)) {
        RECOM_VLOG << "MatchAndRewriteMatMul successfully";
      } else if (MatchAndRecordSelect(to_dense_node, onode, onode_inid, curr,
                                      context)) {
        RECOM_VLOG << "MatchAndRecordSelect successfully";
      } else if (MatchAndRewriteSoftmax(to_dense_node, onode, onode_inid, curr,
                                        context)) {
        RECOM_VLOG << "MatchAndRewriteSoftmax successfully";
      } else if (MatchAndRewriteMul(to_dense_node, onode, onode_inid, curr,
                                    context)) {
        RECOM_VLOG << "MatchAndRewriteMul successfully";
      } else if (MatchShapeAndReconstruct(to_dense_node, onode, onode_inid,
                                          curr, context)) {
        RECOM_VLOG << "MatchShapeAndReconstruct successfully";
        return true;
      } else if (ReconstructToDense(to_dense_node, onode, onode_inid, curr,
                                    context)) {
        RECOM_VLOG << "ReconstructToDense";
        to_dense_node = nullptr;
      }
    }
  }

  const HashSetT<std::string> next_onames = out_mapping.at(onode->name());
  std::vector<NodeDef *> to_dense_node_copies(next_onames.size(),
                                              to_dense_node);
  if (to_dense_node) {
    for (int i = 1; i < to_dense_node_copies.size(); ++i) {
      NodeDef *copy = gd->add_node();
      copy->CopyFrom(*to_dense_node);
      copy->set_name(to_dense_node->name() + "_" +
                     std::to_string(context.to_dense_cpy_cnt++));
      to_dense_node_copies[i] = copy;
    }
  }

  bool success = true;
  auto copy_itr = to_dense_node_copies.begin();
  for (const std::string &next_oname : next_onames) {
    NodeDef *next_onode = node_mapping.at(next_oname);
    int next_onode_inid;
    for (next_onode_inid = 0; next_onode_inid < next_onode->input_size();
         ++next_onode_inid) {
      if (next_onode->input(next_onode_inid) == onode->name())
        break;
    }
    RETURN_IF_FALSE(next_onode_inid < next_onode->input_size());
    success = success && Optimize(*(copy_itr++), next_onode, next_onode_inid,
                                  curr, context);
  }

  return success;
}

bool PostLookupOptimizer::MatchAndRecordReshape(NodeDef *to_dense_node,
                                                NodeDef *onode, int onode_inid,
                                                ExSpInfo &curr,
                                                PostGraphContext &context) {
  RETURN_IF_FALSE(to_dense_node && onode);

  if (onode->op() == "Reshape") {
    RETURN_IF_FALSE(onode_inid == 0);

    RETURN_IF_FALSE(symbolic_context->ContentKnown(onode->input(1)) &&
                    symbolic_context->ShapeKnown(onode->name()));
    curr.dense_shape = symbolic_context->GetShape(onode->name());
  } else if (onode->op() == "Squeeze") {
    RETURN_IF_FALSE(symbolic_context->ShapeKnown(onode->name()));
    curr.dense_shape = symbolic_context->GetShape(onode->name());
  } else if (onode->op() == "Transpose") {
    RETURN_IF_FALSE(onode_inid == 0);

    std::vector<int> perm;
    RETURN_IF_FALSE(symbolic_context->ContentStatic(onode->input(1), perm));

    for (int i = 0; i < perm.size(); ++i) {
      if (perm[i] != i) {
        LOG_AND_RETURN_FALSE("Transpose useful");
      }
    }
  } else if (onode->op() == "StridedSlice") {
    RETURN_IF_FALSE(onode_inid == 0);

    RETURN_IF_FALSE(symbolic_context->ShapeKnown(onode->input(0)));
    RETURN_IF_FALSE(symbolic_context->ShapeKnown(onode->name()));
    const ExprVec input_shape = symbolic_context->GetShape(onode->input(0));
    const ExprVec output_shape = symbolic_context->GetShape(onode->name());
    RETURN_IF_FALSE(symbolic_context->IsEq(
        std::accumulate(input_shape.begin(), input_shape.end(), Expression(1),
                        std::multiplies<Expression>()),
        std::accumulate(output_shape.begin(), output_shape.end(), Expression(1),
                        std::multiplies<Expression>())))
    curr.dense_shape = output_shape;
  } else {
    return false;
  }

  return true;
}

bool PostLookupOptimizer::MatchAndRewriteMatMul(NodeDef *to_dense_node,
                                                NodeDef *onode, int onode_inid,
                                                ExSpInfo &curr,
                                                PostGraphContext &context) {
  RETURN_IF_FALSE(to_dense_node && onode);

  if (onode->op() == "MatMul") {
    RETURN_IF_FALSE(onode_inid == 0 || onode_inid == 1);
    RETURN_IF_FALSE(curr.default_value == 0);

    RETURN_IF_FALSE(symbolic_context->ShapeKnown(onode->input(1 - onode_inid)));
    const ExprVec b_shape =
        symbolic_context->GetShape(onode->input(1 - onode_inid));
    assert(curr.dense_shape.size() == 2);
    assert(b_shape.size() == 2);

    bool transpose_a = onode->attr().at("transpose_a").b();
    bool transpose_b = onode->attr().at("transpose_b").b();
    if (onode_inid == 1)
      std::swap(transpose_a, transpose_b);
    LOG_AND_RETURN_IF_FALSE(
        transpose_a == false,
        "Currently not support ExtendedSparseTensor transpose for MatMul");

    Expression M = curr.dense_shape[0], K = curr.dense_shape[1];
    Expression N = transpose_b ? b_shape[0] : b_shape[1];

    if (curr.element_size == K) {
      if (curr.value_shape.size() != 2) {
        ExprVec new_value_shape = {curr.value_shape[0], curr.element_size};
        NodeDef *reshape_shape = ConstructShapeNodeByExpr(
            new_value_shape, DT_INT32,
            onode->name() + "_value_reshape/shape_construct");

        NodeDef *reshape = gd->add_node();
        reshape->set_op("Reshape");
        reshape->set_name(onode->name() + "_value_reshape");
        reshape->add_input(to_dense_node->input(1));
        reshape->add_input(reshape_shape->name());
        (*reshape->mutable_attr())["T"].set_type(
            to_dense_node->attr().at("T").type());
        (*reshape->mutable_attr())["Tshape"].set_type(
            reshape_shape->attr().at("T").type());

        *to_dense_node->mutable_input(1) = reshape->name();

        curr.value_shape = new_value_shape;
      }

      NodeDef *matmul = gd->add_node();
      matmul->set_op("MatMul");
      matmul->set_name(onode->name() + "_new");
      matmul->add_input(to_dense_node->input(1));
      matmul->add_input(onode->input(1 - onode_inid));
      (*matmul->mutable_attr())["T"].set_type(onode->attr().at("T").type());
      (*matmul->mutable_attr())["transpose_a"].set_b(transpose_a);
      (*matmul->mutable_attr())["transpose_b"].set_b(transpose_b);

      *to_dense_node->mutable_input(1) = matmul->name();

      curr.value_shape[1] = N;
      curr.dense_shape = {M, N};
      curr.element_size = N;
    } else {
      // TODO
      LOG_AND_RETURN_FALSE(
          "Currently not support rewrite MatMul by BatchMatMul");
    }
  } else if (onode->op() == "BatchMatMul") {
    // TODO: support BatchMatMul
    return false;
  } else if (onode->op() == "BatchMatMulV2") {
    // TODO: support BatchMatMulV2
    return false;
  } else if (onode->op() == "BatchMatMulV3") {
    // TODO: support BatchMatMulV3
    return false;
  } else {
    return false;
  }

  return true;
}

bool PostLookupOptimizer::MatchAndRecordSelect(NodeDef *to_dense_node,
                                               NodeDef *onode, int onode_inid,
                                               ExSpInfo &curr,
                                               PostGraphContext &context) {
  RETURN_IF_FALSE(to_dense_node && onode);

  if (onode->op() == "Select") {
    RETURN_IF_FALSE(onode_inid == 1 || onode_inid == 2);

    /* begin SequenceMask match */
    FIND_INPUT_OR_RETURN(onode, less, 0, "Less");

    FIND_INPUT_OR_RETURN(less, range, 0, "Range");
    INPUT_IS_CONST_OR_RETURN(range, 0, 0, int);
    INPUT_IS_CONST_OR_RETURN(range, 2, 1, int);
    FIND_INPUT_OR_RETURN(range, max, 1, "Max");
    FIND_BI_COMMUT_OP_INPUT_OR_RETURN(max, sequence_len, seqlen_idx,
                                      "ConcatV2");
    INPUT_IS_CONST_OR_RETURN(max, 1 - seqlen_idx, 0, int);

    FIND_INPUT_OR_RETURN(less, cast_or_expand_dims, 1, "Cast,ExpandDims");
    if (cast_or_expand_dims->op() == "Cast") {
      FIND_INPUT_OR_RETURN(cast_or_expand_dims, expand_dims, 0, "ExpandDims");
      cast_or_expand_dims = expand_dims;
    }
    INPUT_IS_CONST_OR_RETURN(cast_or_expand_dims, 1, -1, int);
    FIND_INPUT_OR_RETURN(cast_or_expand_dims, sequence_len_x1, 0, "ConcatV2");
    CONVERGE_OR_RETURN(sequence_len, sequence_len_x1);
    /* end SequenceMask match */

    /* begin SequenceLength match */
    RETURN_IF_FALSE(sequence_len->attr().at("N").i() == 2);
    INPUT_IS_CONST_OR_RETURN(sequence_len, 2, 0, int);

    FIND_INPUT_OR_RETURN(sequence_len, len_cast, 0, "Cast");
    FIND_INPUT_OR_RETURN(len_cast, ceil, 0, "Ceil");
    FIND_INPUT_OR_RETURN(ceil, div_or_cast, 0, "Cast,RealDiv");
    if (div_or_cast->op() == "RealDiv") {
      FIND_INPUT_OR_RETURN(div_or_cast, cast, 0, "Cast");
      INPUT_IS_CONST_OR_RETURN(div_or_cast, 1, 1, int);
      div_or_cast = cast;
    }
    FIND_INPUT_OR_RETURN(div_or_cast, segment_max, 0, "SegmentMax");

    FIND_INPUT_OR_RETURN(segment_max, add, 0, "Add,AddV2");
    FIND_BI_COMMUT_OP_INPUT_OR_RETURN(add, ss1, ss1_idx, "StridedSlice");
    INPUT_IS_SPLAT_CONST_OR_RETURN(add, 1 - ss1_idx, 1, int);

    RETURN_IF_FALSE(ss1->attr().at("begin_mask").i() == 1);
    RETURN_IF_FALSE(ss1->attr().at("ellipsis_mask").i() == 0);
    RETURN_IF_FALSE(ss1->attr().at("end_mask").i() == 1);
    RETURN_IF_FALSE(ss1->attr().at("new_axis_mask").i() == 0);
    RETURN_IF_FALSE(ss1->attr().at("shrink_axis_mask").i() == 2);
    EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(ss1, 1), ss1_begin, int);
    RETURN_IF_FALSE(ss1_begin.size() == 2 && ss1_begin[0] == 0);
    EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(ss1, 3), ss1_stride, int);
    RETURN_IF_FALSE(ss1_stride.size() == 2 && ss1_stride[1] == 1);

    NodeDef *src = GetInputNode(ss1, 0);

    FIND_INPUT_OR_RETURN(segment_max, ss2, 1, "StridedSlice");

    RETURN_IF_FALSE(ss2->attr().at("begin_mask").i() == 1);
    RETURN_IF_FALSE(ss2->attr().at("ellipsis_mask").i() == 0);
    RETURN_IF_FALSE(ss2->attr().at("end_mask").i() == 1);
    RETURN_IF_FALSE(ss2->attr().at("new_axis_mask").i() == 0);
    RETURN_IF_FALSE(ss2->attr().at("shrink_axis_mask").i() == 2);
    EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(ss2, 1), ss2_begin, int);
    RETURN_IF_FALSE(ss2_begin.size() == 2 && ss2_begin[0] == 0);
    EXTRACT_CONST_TENSOR_OR_RETURN(GetInputNode(ss2, 3), ss2_stride, int);
    RETURN_IF_FALSE(ss2_stride.size() == 2 && ss2_stride[1] == 1);

    NodeDef *src_x1 = GetInputNode(ss2, 0);
    CONVERGE_OR_RETURN(src, src_x1);

    // we do not need to check the shape because the subgraph left can ensure it
    INPUT_IS_SPLAT_CONST_OR_RETURN(sequence_len, 1, 0, int);

    /* end SequenceLength match */

    EXTRACT_SPLAT_CONST_OR_RETURN(GetInputNode(onode, 3 - onode_inid),
                                  new_default, float);
    curr.default_value = new_default;
  } else {
    return false;
  }

  return true;
}

bool PostLookupOptimizer::MatchAndRewriteSoftmax(NodeDef *to_dense_node,
                                                 NodeDef *onode, int onode_inid,
                                                 ExSpInfo &curr,
                                                 PostGraphContext &context) {
  RETURN_IF_FALSE(to_dense_node && onode);
  constexpr int THRESHOLD = -10000;

  if (onode->op() == "Softmax") {
    RETURN_IF_FALSE(curr.default_value <= THRESHOLD);
    RETURN_IF_FALSE(curr.element_size == 1);
    assert(curr.dense_shape.size() == 2);

    if (curr.dense_prefix != curr.dense_shape) {
      ExprVec new_dense_prefix = curr.dense_shape;
      NodeDef *new_dense_prefix_node = ConstructShapeNodeByExpr(
          new_dense_prefix, DT_INT64,
          onode->name() + "_indice_reshape/shape_construct");

      NodeDef *sparse_reshape = gd->add_node();
      sparse_reshape->set_op("SparseReshape");
      sparse_reshape->set_name(onode->name() + "_indice_reshape");
      sparse_reshape->add_input(to_dense_node->input(0));
      sparse_reshape->add_input(to_dense_node->input(2));
      sparse_reshape->add_input(new_dense_prefix_node->name());

      *to_dense_node->mutable_input(0) = sparse_reshape->name();
      *to_dense_node->mutable_input(2) = new_dense_prefix_node->name();

      curr.dense_prefix = new_dense_prefix;
    }

    if (curr.value_shape.size() != 1) {
      NodeDef *negative_one = gd->add_node();
      negative_one->set_op("Const");
      negative_one->set_name(onode->name() + "_value_reshape/const");
      AttrValue negative_one_value;
      negative_one_value.mutable_tensor()->set_dtype(DT_INT32);
      negative_one_value.mutable_tensor()
          ->mutable_tensor_shape()
          ->add_dim()
          ->set_size(1);
      negative_one_value.mutable_tensor()->add_int_val(-1);
      (*negative_one->mutable_attr())["value"] = negative_one_value;
      (*negative_one->mutable_attr())["dtype"].set_type(DT_INT32);

      NodeDef *reshape = gd->add_node();
      reshape->set_op("Reshape");
      reshape->set_name(onode->name() + "_value_reshape");
      reshape->add_input(to_dense_node->input(1));
      reshape->add_input(negative_one->name());
      (*reshape->mutable_attr())["T"].set_type(
          to_dense_node->attr().at("T").type());
      (*reshape->mutable_attr())["Tshape"].set_type(
          negative_one->attr().at("dtype").type());

      *to_dense_node->mutable_input(1) = reshape->name();

      curr.value_shape = {curr.value_shape[0]};
    }

    NodeDef *sparse_softmax = gd->add_node();
    sparse_softmax->set_op("SparseSoftmax");
    sparse_softmax->set_name(onode->name() + "_new");
    sparse_softmax->add_input(to_dense_node->input(0));
    sparse_softmax->add_input(to_dense_node->input(1));
    sparse_softmax->add_input(to_dense_node->input(2));
    (*sparse_softmax->mutable_attr())["T"].set_type(
        to_dense_node->attr().at("T").type());

    *to_dense_node->mutable_input(1) = sparse_softmax->name();

    curr.default_value = 0;
  } else {
    return false;
  }

  return true;
}

bool PostLookupOptimizer::MatchAndRewriteMul(NodeDef *to_dense_node,
                                             NodeDef *onode, int onode_inid,
                                             ExSpInfo &curr,
                                             PostGraphContext &context) {
  RETURN_IF_FALSE(to_dense_node && onode);

  if (onode->op() == "Mul") {
    if (context.pending_nodes.count(onode->name())) {
      assert(context.pending_nodes.at(onode->name()).size() == 1);
      auto pending_pair =
          context.pending_nodes.at(onode->name()).at(1 - onode_inid);
      NodeDef *pending_to_dense = pending_pair.first;
      ExSpInfo pending_info = pending_pair.second;

      ExprVec pending_expect_prefix, pending_expect_value_shape;
      RETURN_IF_FALSE(GetExpectDensePrefixAndValueShape(
          pending_info, pending_expect_prefix, pending_expect_value_shape));

      ExprVec curr_expect_prefix, curr_expect_value_shape;
      RETURN_IF_FALSE(GetExpectDensePrefixAndValueShape(
          curr, curr_expect_prefix, curr_expect_value_shape));

      LOG_AND_RETURN_IF_FALSE(
          symbolic_context->IsEq(pending_expect_prefix, curr_expect_prefix),
          "Currently do not support prefix broadcast for Mul");

      RETURN_IF_FALSE(curr.default_value == 0 &&
                      pending_info.default_value == 0);

      if (!symbolic_context->IsEq(curr.value_shape, curr_expect_value_shape)) {
        NodeDef *reshape_shape = ConstructShapeNodeByExpr(
            curr_expect_value_shape, DT_INT32,
            onode->name() + "_value_reshape/shape_construct_1");

        NodeDef *reshape = gd->add_node();
        reshape->set_op("Reshape");
        reshape->set_name(onode->name() + "_value_reshape_1");
        reshape->add_input(to_dense_node->input(1));
        reshape->add_input(reshape_shape->name());
        (*reshape->mutable_attr())["T"].set_type(
            to_dense_node->attr().at("T").type());
        (*reshape->mutable_attr())["Tshape"].set_type(
            reshape_shape->attr().at("T").type());

        *to_dense_node->mutable_input(1) = reshape->name();

        curr.value_shape = curr_expect_value_shape;
      }

      if (!symbolic_context->IsEq(pending_info.value_shape,
                                  pending_expect_value_shape)) {
        NodeDef *reshape_shape = ConstructShapeNodeByExpr(
            pending_expect_value_shape, DT_INT32,
            onode->name() + "_value_reshape/shape_construct_2");

        NodeDef *reshape = gd->add_node();
        reshape->set_op("Reshape");
        reshape->set_name(onode->name() + "_value_reshape_2");
        reshape->add_input(pending_to_dense->input(1));
        reshape->add_input(reshape_shape->name());
        (*reshape->mutable_attr())["T"].set_type(
            pending_to_dense->attr().at("T").type());
        (*reshape->mutable_attr())["Tshape"].set_type(
            reshape_shape->attr().at("T").type());

        *pending_to_dense->mutable_input(1) = reshape->name();

        pending_info.value_shape = pending_expect_value_shape;
      }

      NodeDef *mul = gd->add_node();
      mul->set_op("Mul");
      mul->set_name(onode->name() + "_new");
      mul->add_input(to_dense_node->input(1));
      mul->add_input(pending_to_dense->input(1));
      (*mul->mutable_attr())["T"].set_type(onode->attr().at("T").type());

      RETURN_IF_FALSE(symbolic_context->ShapeKnown(onode->name()));
      ExprVec new_dense_shape = symbolic_context->GetShape(onode->name());
      ExprVec new_value_shape(new_dense_shape.size() -
                              curr_expect_prefix.size() + 1);
      new_value_shape[0] = curr.value_shape[0];
      std::copy(new_dense_shape.cbegin() + curr_expect_prefix.size(),
                new_dense_shape.cend(), new_value_shape.begin() + 1);
      Expression new_element_size =
          std::accumulate(new_value_shape.cbegin() + 1, new_value_shape.cend(),
                          Expression(1), std::multiplies<Expression>());
      curr.dense_shape = new_dense_shape;
      curr.value_shape = new_value_shape;
      curr.element_size = new_element_size;

      RECOM_VLOG << ExprVecToStr(curr.dense_shape);

      *to_dense_node->mutable_input(1) = mul->name();
    } else { // TODO: support more cases
      if (onode->attr().at("T").type() == DT_FLOAT) {
        EXTRACT_SPLAT_CONST_OR_RETURN(onode, c, float);

        NodeDef *const_node = gd->add_node();
        const_node->set_op("Const");
        const_node->set_name(onode->name() + "_new/const");
        AttrValue const_value;
        const_value.mutable_tensor()->set_dtype(DT_FLOAT);
        const_value.mutable_tensor()
            ->mutable_tensor_shape()
            ->add_dim()
            ->set_size(1);
        const_value.mutable_tensor()->add_float_val(c);
        (*const_node->mutable_attr())["value"] = const_value;
        (*const_node->mutable_attr())["dtype"].set_type(DT_FLOAT);

        NodeDef *mul = gd->add_node();
        mul->set_op("Mul");
        mul->set_name(onode->name() + "_new");
        mul->add_input(to_dense_node->input(1));
        mul->add_input(const_node->name());
        (*mul->mutable_attr())["T"].set_type(DT_FLOAT);

        *to_dense_node->mutable_input(1) = mul->name();

        curr.default_value *= c;
      }
    }
  } else {
    return false;
  }

  return true;
}

bool PostLookupOptimizer::MatchShapeAndReconstruct(NodeDef *to_dense_node,
                                                   NodeDef *onode,
                                                   int onode_inid,
                                                   ExSpInfo &curr,
                                                   PostGraphContext &context) {
  RETURN_IF_FALSE(to_dense_node && onode);

  if (onode->op() == "Shape") {
    NodeDef *shape_construct = ConstructShapeNodeByExpr(
        curr.dense_shape, onode->attr().at("out_type").type(),
        onode->name() + "_shape_construct");
    for (const std::string &next_name : out_mapping.at(onode->name())) {
      NodeDef *node = node_mapping.at(next_name);
      for (int i = 0; i < node->input_size(); ++i) {
        if (node->input(i) == onode->name()) {
          *node->mutable_input(i) = shape_construct->name();
        }
      }
    }
  } else if (onode->op() == "ZerosLike" || onode->op() == "OnesLike") {
    // TODO
    LOG_AND_RETURN_FALSE("Currently not support " + onode->op() +
                         " reconstruct");
  } else {
    return false;
  }

  return true;
}

bool PostLookupOptimizer::ReconstructToDense(NodeDef *to_dense_node,
                                             NodeDef *onode, int onode_inid,
                                             ExSpInfo &curr,
                                             PostGraphContext &context) {
  RETURN_IF_FALSE(to_dense_node && onode);

  (*to_dense_node->mutable_attr())["default_float"].set_f(curr.default_value);

  ExprVec real_dense_shape = curr.dense_prefix;
  real_dense_shape.insert(real_dense_shape.end(), curr.value_shape.begin() + 1,
                          curr.value_shape.end());

  if (real_dense_shape == curr.dense_shape) {
    (*onode->mutable_input(onode_inid)) = to_dense_node->name();
  } else {
    NodeDef *reshape_shape = ConstructShapeNodeByExpr(
        curr.dense_shape, DT_INT32,
        to_dense_node->name() + "_reshape/shape_construct");

    NodeDef *reshape = gd->add_node();
    reshape->set_op("Reshape");
    reshape->set_name(to_dense_node->name() + "_reshape");
    reshape->add_input(to_dense_node->name());
    reshape->add_input(reshape_shape->name());
    (*reshape->mutable_attr())["T"].set_type(
        to_dense_node->attr().at("T").type());
    (*reshape->mutable_attr())["Tshape"].set_type(
        reshape_shape->attr().at("T").type());

    (*onode->mutable_input(onode_inid)) = reshape->name();
  }

  return true;
}

} // namespace feature_opt
} // namespace tensorflow