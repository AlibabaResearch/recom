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

#include "ceil_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include <symengine/functions.h>
#include <tensorflow/core/framework/types.h>

namespace tensorflow {
namespace feature_opt {

bool CeilOpInferFn::InferCeilShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  context->SetShape(node->name(), context->GetShape(node->input(0)));
  return true;
}

bool CeilOpInferFn::InferCeilContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->InputContentKnown(node));
  ExprVec content = context->GetContent(node->input(0));
  for (auto &expr : content) {
    expr = Ceiling(expr);
  }

  context->SetContent(node->name(), content);
  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Ceil", CeilOpInferFn);

} // namespace feature_opt
} // namespace tensorflow