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

#include "arithm_ops_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include <functional>
#include <numeric>
#include <symengine/number.h>
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool ComputeBroadCastContent(const std::vector<int> &input_shape,
                             const std::vector<int> &output_shape,
                             const ExprVec &input, ExprVec &output) {
  std::vector<int> padded_input_shape(output_shape.size());
  auto padded_itr = padded_input_shape.rbegin();
  for (auto input_itr = input_shape.crbegin(); input_itr < input_shape.crend();
       ++input_itr, ++padded_itr) {
    *padded_itr = *input_itr;
  }

  while (padded_itr != padded_input_shape.rend()) {
    *padded_itr = 1;
    ++padded_itr;
  }

  const int output_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  output = ExprVec(output_size);

  const std::vector<int> output_offsets = ComputeOffsetUnits(output_shape, 1);
  const std::vector<int> padded_offsets =
      ComputeOffsetUnits(padded_input_shape, 1);
  for (int output_idx = 0; output_idx < output_size; ++output_idx) {
    int n = output_idx;
    int input_idx = 0;
    for (int i = 0; i < output_shape.size(); ++i) {
      int idx = n / output_offsets[i];
      input_idx += std::min(idx, padded_input_shape[i] - 1) * padded_offsets[i];
      n -= idx * output_offsets[i];
    }
    output[output_idx] = input[input_idx];
  }

  return true;
}

template <class ArithmOp>
bool ArithmOpInferFn<ArithmOp>::InferShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  const ExprVec a_shape = context->GetShape(node->input(0));
  const ExprVec b_shape = context->GetShape(node->input(1));

  ExprVec output_shape;
  RETURN_IF_FALSE(GetBroadCastShape(context, a_shape, b_shape, output_shape));

  context->SetShape(node->name(), output_shape);

  return true;
}

template <class ArithmOp>
bool ArithmOpInferFn<ArithmOp>::InferContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->InputContentKnown(node));
  const ExprVec &a = context->GetContent(node->input(0));
  const ExprVec &b = context->GetContent(node->input(1));

  std::vector<int> output_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->name(), output_shape));

  std::vector<int> a_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->input(0), a_shape));
  ExprVec a_broadcast;
  RETURN_IF_FALSE(
      ComputeBroadCastContent(a_shape, output_shape, a, a_broadcast));

  std::vector<int> b_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->input(1), b_shape));
  ExprVec b_broadcast;
  RETURN_IF_FALSE(
      ComputeBroadCastContent(b_shape, output_shape, b, b_broadcast));

  ExprVec output(a_broadcast.size());
  for (int i = 0; i < output.size(); ++i) {
    output[i] = ArithmOp()(a_broadcast[i], b_broadcast[i]);
  }

  context->SetContent(node->name(), output);

  return true;
}

template class ArithmOpInferFn<std::plus<Expression>>;
template class ArithmOpInferFn<std::multiplies<Expression>>;
template class ArithmOpInferFn<std::minus<Expression>>;
template class ArithmOpInferFn<std::divides<Expression>>;

REGISTER_SYMBOLIC_SHAPE_FN("Add", ArithmOpInferFn<std::plus<Expression>>);
REGISTER_SYMBOLIC_SHAPE_FN("AddV2", ArithmOpInferFn<std::plus<Expression>>);
REGISTER_SYMBOLIC_SHAPE_FN("Mul", ArithmOpInferFn<std::multiplies<Expression>>);
REGISTER_SYMBOLIC_SHAPE_FN("Sub", ArithmOpInferFn<std::minus<Expression>>);
REGISTER_SYMBOLIC_SHAPE_FN("RealDiv",
                           ArithmOpInferFn<std::divides<Expression>>);

} // namespace feature_opt
} // namespace tensorflow