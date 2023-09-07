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

#include "expand_dims_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"

namespace tensorflow {
namespace feature_opt {

bool ExpandDimsOpInferFn::InferExpandDimsShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  ExprVec shape = context->GetShape(node->input(0));

  LOG_AND_RETURN_IF_FALSE(context->ContentStatic(node->input(1)),
                          "Not support dynamic axis for Concat op!");
  int axis = static_cast<int>(context->GetContent(node->input(1))[0]);
  if (axis < 0)
    axis += shape.size() + 1;

  shape.insert(shape.begin() + axis, 1);

  context->SetShape(node->name(), shape);

  return true;
}

bool ExpandDimsOpInferFn::InferExpandDimsContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  context->SetContent(node->name(), context->GetContent(node->input(0)));
  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("ExpandDims", ExpandDimsOpInferFn);

} // namespace feature_opt
} // namespace tensorflow