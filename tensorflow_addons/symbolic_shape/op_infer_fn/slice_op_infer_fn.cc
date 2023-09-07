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

#include "slice_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"

namespace tensorflow {
namespace feature_opt {

bool SliceOpInferFn::InferSliceShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  if (context->ContentKnown(node->input(2))) {
    RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
    const ExprVec input_shape = context->GetShape(node->input(0));

    ExprVec size = context->GetContent(node->input(2));
    assert(input_shape.size() == size.size());

    for (int i = 0; i < size.size(); ++i) {
      if (size[i] == Expression(-1)) {
        size[i] = input_shape[i] - 1;
      }
    }

    context->SetShape(node->name(), size);
  } else {
    RETURN_IF_FALSE(context->ShapeKnown(node->input(2)));
    RECOM_VLOG << "Assign new symbols to Slice op output shape";

    ExprVec size_shape = context->GetShape(node->input(2));
    ExprVec output_shape(size_shape.size());
    std::generate(output_shape.begin(), output_shape.end(),
                  [&] { return context->AddNewSymbol(node); });
    context->SetShape(node->name(), output_shape);
  }

  return true;
}

bool SliceOpInferFn::InferSliceContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->InputContentKnown(node));

  ExprVec input = context->GetContent(node->input(0));

  std::vector<int> input_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->input(0), input_shape));

  std::vector<int> begin;
  RETURN_IF_FALSE(context->ContentStatic(node->input(1), begin));

  std::vector<int> size;
  RETURN_IF_FALSE(context->ContentStatic(node->input(2), size));
  for (int i = 0; i < size.size(); ++i) {
    if (size[i] == -1) {
      size[i] = input_shape[i] - 1;
    }
  }
  int output_size =
      std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());

  const std::vector<int> offset_units = ComputeOffsetUnits(input_shape, 1);

  ExprVec output(output_size, 1);
  auto itr = output.begin();
  std::function<bool(int, int)> slice = [&](int i, int offset) -> bool {
    const int begin_idx = begin[i];
    const int end_idx = begin_idx + size[i];
    if (i + 1 == input_shape.size()) {
      for (int idx = begin_idx; idx < end_idx; ++idx) {
        *(itr++) = input[offset + idx];
      }
    } else {
      for (int idx = begin_idx; idx < end_idx; ++idx) {
        RETURN_IF_FALSE(slice(i + 1, offset + idx * offset_units[i]));
      }
    }
    return true;
  };
  slice(0, 0);

  context->SetContent(node->name(), output);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Slice", SliceOpInferFn);

} // namespace feature_opt
} // namespace tensorflow