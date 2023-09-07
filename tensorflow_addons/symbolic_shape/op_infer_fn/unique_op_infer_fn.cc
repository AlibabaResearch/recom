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

#include "unique_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool UniqueOpInferFn::InferUniqueShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  const ExprVec input_shape = context->GetShape(node->input(0));
  assert(input_shape.size() == 1);

  context->SetShape(node->name(), {context->AddNewSymbol(node)});
  context->SetShape(FormTensorName(node, 1), input_shape);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Unique", UniqueOpInferFn);

} // namespace feature_opt
} // namespace tensorflow