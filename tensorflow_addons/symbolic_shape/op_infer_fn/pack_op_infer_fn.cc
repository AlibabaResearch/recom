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

#include "pack_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <functional>
#include <numeric>
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool PackOpInferFn::ConstrainInput(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  const int N = node->attr().at("N").i();
  bool ret = true;
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      const ExprVec input_i = context->GetShape(node->input(i));
      const ExprVec input_j = context->GetShape(node->input(j));
      LOG_AND_RETURN_IF_FALSE(
          input_i.size() == input_j.size(),
          "Input tensors of Pack op are not of same shape!");
      for (int k = 0; k < input_i.size(); ++k) {
        if (!context->IsEq(input_i[k], input_j[k])) {
          ret = ret && context->MakeEq(input_i[k], input_j[k]);
        }
      }
    }
  }

  return ret;
}

bool PackOpInferFn::InferPackShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  const int N = node->attr().at("N").i();
  int axis = node->attr().at("axis").i();

  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  ExprVec shape = context->GetShape(node->input(0));
  if (axis < 0)
    axis += shape.size();
  shape.insert(shape.begin() + axis, N);
  context->SetShape(node->name(), shape);

  return true;
}

bool PackOpInferFn::InferPackContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->InputContentKnown(node));
  const int N = node->attr().at("N").i();
  int axis = node->attr().at("axis").i();

  std::vector<int> input_shape;
  LOG_AND_RETURN_IF_FALSE(context->ShapeStatic(node->input(0), input_shape),
                          "Currently only support static shape input for Pack "
                          "op content inference");
  if (axis < 0)
    axis += input_shape.size();

  const int pre_axis_prod =
      std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1,
                      std::multiplies<int>());
  const int post_axis_prod = std::accumulate(
      input_shape.begin() + axis, input_shape.end(), 1, std::multiplies<int>());
  ExprVec output(pre_axis_prod * N * post_axis_prod);
  for (int i = 0; i < N; ++i) {
    const ExprVec input = context->GetContent(node->input(i));
    for (int input_idx = 0; input_idx < input.size(); ++input_idx) {
      const int pre_idx = input_idx / post_axis_prod;
      const int post_idx = input_idx % post_axis_prod;
      const int output_idx =
          pre_idx * N * post_axis_prod + i * post_axis_prod + post_idx;
      output[output_idx] = input[input_idx];
    }
  }

  context->SetContent(node->name(), output);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Pack", PackOpInferFn);

} // namespace feature_opt
} // namespace tensorflow