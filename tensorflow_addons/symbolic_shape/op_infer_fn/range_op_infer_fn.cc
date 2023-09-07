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

#include <numeric>

#include "range_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"

namespace tensorflow {
namespace feature_opt {

bool RangeOpInferFn::InferRangeShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->InputContentKnown(node));
  const Expression start = context->GetContent(node->input(0))[0];
  const Expression limit = context->GetContent(node->input(1))[0];
  const Expression delta = context->GetContent(node->input(2))[0];
  const Expression num_output = Ceiling((limit - start) / delta);

  context->SetShape(node->name(), {num_output});

  return true;
}

bool RangeOpInferFn::InferRangeContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  std::vector<int> output_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->name(), output_shape));
  const int num_output = output_shape[0];

  ExprVec output(num_output);
  for (int i = 0; i < num_output; ++i) {
    output[i] = Expression(i);
  }

  context->SetContent(node->name(), output);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Range", RangeOpInferFn);

} // namespace feature_opt
} // namespace tensorflow