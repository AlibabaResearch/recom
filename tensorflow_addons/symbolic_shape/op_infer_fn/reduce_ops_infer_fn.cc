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

#include "reduce_ops_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <functional>
#include <numeric>
#include <symengine/expression.h>
#include <vector>

namespace tensorflow {
namespace feature_opt {

template <class ReduceOp, int Init>
bool ReduceOpInferFn<ReduceOp, Init>::InferReduceShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  if (node->input_size() == 1) { // axis = None
    context->SetShape(node->name(), {Expression(1)});
    return true;
  }

  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  const ExprVec input_shape = context->GetShape(node->input(0));

  std::vector<int> axes;
  LOG_AND_RETURN_IF_FALSE(context->ContentStatic(node->input(1), axes),
                          "Not suport dynamic axis for Reduce op");
  for (int &axis : axes) {
    if (axis < 0)
      axis += input_shape.size();
  }

  const bool keep_dims = node->attr().at("keep_dims").b();
  ExprVec output_shape;
  if (keep_dims) {
    output_shape = input_shape;
    for (int axis : axes) {
      output_shape[axis] = Expression(1);
    }
  } else {
    std::vector<bool> flags(input_shape.size(), true);
    for (int axis : axes) {
      flags[axis] = false;
    }

    for (int i = 0; i < input_shape.size(); ++i) {
      if (flags[i])
        output_shape.push_back(input_shape[i]);
    }
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

template <class ReduceOp, int Init>
bool ReduceOpInferFn<ReduceOp, Init>::InferReduceContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  ExprVec input = context->GetContent(node->input(0));
  if (node->input_size() == 1) { // axis = None
    Expression res = std::accumulate(input.begin(), input.end(),
                                     Expression(Init), ReduceOp());
    context->SetContent(node->name(), {res});
    return true;
  }

  std::vector<int> input_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->input(0), input_shape));
  const int shape_len = input_shape.size();

  std::vector<int> axes;
  LOG_AND_RETURN_IF_FALSE(context->ContentStatic(node->input(1), axes),
                          "Not suport dynamic axis for Reduce op");
  for (int &axis : axes) {
    if (axis < 0)
      axis += shape_len;
  }

  std::vector<int> kept_output_shape = input_shape;
  std::vector<bool> flags(shape_len, true);
  for (int axis : axes) {
    flags[axis] = false;
    kept_output_shape[axis] = 1;
  }

  const std::vector<int> input_offset_units =
      ComputeOffsetUnits(input_shape, 1);
  const std::vector<int> output_offset_units =
      ComputeOffsetUnits(kept_output_shape, 1);

  const int num_output =
      (flags[0] ? input_shape[0] : 1) * output_offset_units[0];
  ExprVec output(num_output, Expression(Init));
  for (int i = 0; i < input.size(); ++i) {
    int input_idx = i;
    int output_idx = 0;
    for (int j = 0; j < shape_len; ++j) {
      int idx = input_idx / input_offset_units[j];
      if (flags[j]) {
        output_idx += idx * output_offset_units[j];
      }
      input_idx -= idx * input_offset_units[j];
    }
    output[output_idx] = ReduceOp()(output[output_idx], input[i]);
  }

  context->SetContent(node->name(), output);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Prod",
                           ReduceOpInferFn<std::multiplies<Expression>, 1>);
REGISTER_SYMBOLIC_SHAPE_FN("Sum",
                           ReduceOpInferFn<std::plus<Expression>, 0>);

} // namespace feature_opt
} // namespace tensorflow