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

#include "concat_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <functional>
#include <numeric>
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool ConcatOpInferFn::ConstrainInput(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  const int N = node->attr().at("N").i();
  for (int i = 0; i < N; ++i) {
    RETURN_IF_FALSE(context->ShapeKnown(node->input(i)));
  }
  const int rank = context->GetShape(node->input(0)).size();

  LOG_AND_RETURN_IF_FALSE(context->ContentStatic(node->input(N)),
                          "Not support dynamic axis for Concat op!");
  int axis = static_cast<int>(context->GetContent(node->input(N))[0]);
  if (axis < 0)
    axis += rank;

  bool ret = true;
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      const ExprVec input_i = context->GetShape(node->input(i));
      const ExprVec input_j = context->GetShape(node->input(j));
      assert(input_i.size() == input_j.size());
      for (int k = 0; k < input_i.size(); ++k) {
        if (k != axis && !context->IsEq(input_i[k], input_j[k])) {
          ret = ret && context->MakeEq(input_i[k], input_j[k]);
        }
      }
    }
  }

  return ret;
}

bool ConcatOpInferFn::InferConcatShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  const int N = node->attr().at("N").i();
  for (int i = 0; i < N; ++i) {
    RETURN_IF_FALSE(context->ShapeKnown(node->input(i)));
  }
  ExprVec output_shape = context->GetShape(node->input(0));
  const int rank = output_shape.size();

  LOG_AND_RETURN_IF_FALSE(context->ContentStatic(node->input(N)),
                          "Not support dynamic axis for Concat op!");
  int axis = static_cast<int>(context->GetContent(node->input(N))[0]);
  if (axis < 0)
    axis += rank;

  for (int i = 1; i < N; ++i) {
    ExprVec input_shape_i = context->GetShape(node->input(i));
    output_shape[axis] += input_shape_i[axis];
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

bool ConcatOpInferFn::InferConcatContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->InputContentKnown(node));
  const int N = node->attr().at("N").i();

  std::vector<int> output_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->name(), output_shape));
  const int rank = output_shape.size();

  RETURN_IF_FALSE(context->ContentStatic(node->input(N)));
  int axis = static_cast<int>(context->GetContent(node->input(N))[0]);
  if (axis < 0)
    axis += rank;

  const int pre_axis_prod =
      std::accumulate(output_shape.begin(), output_shape.begin() + axis, 1,
                      std::multiplies<int>());
  const int post_axis_prod =
      std::accumulate(output_shape.begin() + axis + 1, output_shape.end(), 1,
                      std::multiplies<int>());
  const int output_offset_unit_at_axis = output_shape[axis] * post_axis_prod;
  const int output_size = pre_axis_prod * output_offset_unit_at_axis;
  ExprVec output(output_size);

  int prefix_at_axis = 0;
  for (int i = 0; i < N; ++i) {
    std::vector<int> input_shape_i;
    RETURN_IF_FALSE(context->ShapeStatic(node->input(i), input_shape_i));
    const int shape_at_axis = input_shape_i[axis];

    const ExprVec input_i = context->GetContent(node->input(i));
    auto input_itr = input_i.cbegin();
    for (int pre_idx = 0; pre_idx < pre_axis_prod; ++pre_idx) {
      for (int idx_at_axis = 0; idx_at_axis < shape_at_axis; ++idx_at_axis) {
        for (int post_idx = 0; post_idx < post_axis_prod; ++post_idx) {
          const int output_idx =
              pre_idx * output_offset_unit_at_axis +
              (prefix_at_axis + idx_at_axis) * post_axis_prod + post_idx;
          output[output_idx] = *(input_itr++);
        }
      }
    }
    prefix_at_axis += shape_at_axis;
  }

  context->SetContent(node->name(), output);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("ConcatV2", ConcatOpInferFn);

} // namespace feature_opt
} // namespace tensorflow