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

#include "transpose_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool TransposeOpInferFn::InferTransposeShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  const ExprVec input_shape = context->GetShape(node->input(0));

  std::vector<int> perm;
  if (node->input_size() > 1) {
    LOG_AND_RETURN_IF_FALSE(context->ContentStatic(node->input(1), perm),
                            "Not support dynamic perm of Transpose op!");
    assert(perm.size() == input_shape.size());
  } else {
    perm = std::vector<int>(input_shape.size());
    std::iota(perm.rbegin(), perm.rend(), 0);
  }

  ExprVec output_shape(input_shape.size());
  for (int i = 0; i < perm.size(); ++i) {
    output_shape[i] = input_shape[perm[i]];
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

bool TransposeOpInferFn::InferTransposeContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  const ExprVec input = context->GetContent(node->input(0));

  std::vector<int> input_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->input(0), input_shape));
  const std::vector<int> input_offset_units =
      ComputeOffsetUnits(input_shape, 1);

  std::vector<int> perm;
  if (node->input_size() > 1) {
    LOG_AND_RETURN_IF_FALSE(context->ContentStatic(node->input(1), perm),
                            "Not support dynamic perm of Transpose op!");
    assert(perm.size() == input_shape.size());
  } else {
    perm = std::vector<int>(input_shape.size());
    std::iota(perm.rbegin(), perm.rend(), 0);
  }

  std::vector<int> output_shape(input_shape.size());
  for (int i = 0; i < perm.size(); ++i) {
    output_shape[i] = input_shape[perm[i]];
  }
  const std::vector<int> output_offset_units =
      ComputeOffsetUnits(output_shape, 1);

  ExprVec output(input.size());
  for (int i = 0; i < input.size(); ++i) {
    int input_idx = i;
    int output_idx = 0;
    for (int j = 0; j < input_shape.size(); ++j) {
      int idx = input_idx / input_offset_units[j];
      output_idx += idx * output_offset_units[perm[j]];
      input_idx -= idx * input_offset_units[j];
    }
    output[output_idx] = input[i];
  }

  context->SetContent(node->name(), output);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Transpose", TransposeOpInferFn);

} // namespace feature_opt
} // namespace tensorflow