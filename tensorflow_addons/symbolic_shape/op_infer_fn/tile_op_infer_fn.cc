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

#include "tile_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include <functional>
#include <numeric>
#include <symengine/expression.h>
#include <symengine/number.h>
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool TileOpInferFn::InferTileShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  const ExprVec input_shape = context->GetShape(node->input(0));

  RETURN_IF_FALSE(context->ContentKnown(node->input(1)));
  const ExprVec multiples = context->GetContent(node->input(1));
  assert(input_shape.size() == multiples.size());

  ExprVec output_shape(input_shape.size());
  for (int i = 0; i < output_shape.size(); ++i) {
    output_shape[i] = input_shape[i] * multiples[i];
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

bool TileOpInferFn::InferTileContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  std::vector<int> input_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->input(0), input_shape));
  std::vector<int> input_offset_units = ComputeOffsetUnits(input_shape, 1);

  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  const ExprVec input = context->GetContent(node->input(0));

  std::vector<int> output_shape;
  RETURN_IF_FALSE(context->ContentStatic(node->name(), output_shape));
  assert(input_shape.size() == output_shape.size());
  std::vector<int> output_offset_units = ComputeOffsetUnits(output_shape, 1);
  const int output_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  ExprVec output(output_size);
  for (int i = 0; i < output_size; ++i) {
    int output_idx = i;
    int input_idx = 0;
    for (int j = 0; j < output_shape.size(); ++j) {
      int idx = output_idx / output_offset_units[j];
      input_idx += idx * input_offset_units[j];
      output_idx -= idx * output_offset_units[j];
    }
    output[i] = input[input_idx];
  }

  context->SetContent(node->name(), output);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Tile", TileOpInferFn);

} // namespace feature_opt
} // namespace tensorflow