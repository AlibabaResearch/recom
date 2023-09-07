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

#include "select_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool SelectOpInferFn::ConstrainInput(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(2)));

  bool ret = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      const ExprVec input_i = context->GetShape(node->input(i));
      const ExprVec input_j = context->GetShape(node->input(j));
      assert(input_i.size() == input_j.size());
      for (int k = 0; k < input_i.size(); ++k) {
        if (!context->IsEq(input_i[k], input_j[k])) {
          ret = ret && context->MakeEq(input_i[k], input_j[k]);
        }
      }
    }
  }

  return ret;
}

bool SelectOpInferFn::InferSelectShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  context->SetShape(node->name(), context->GetShape(node->input(0)));

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Select", SelectOpInferFn);

} // namespace feature_opt
} // namespace tensorflow