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

#include "fill_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <functional>
#include <numeric>
#include <symengine/expression.h>
#include <symengine/number.h>
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool FillOpInferFn::InferFillShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  context->SetShape(node->name(), context->GetContent(node->input(0)));

  return true;
}

bool FillOpInferFn::InferFillContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->InputContentKnown(node));
  const ExprVec dims = context->GetContent(node->input(0));
  const ExprVec value = context->GetContent(node->input(1));
  assert(value.size() == 1);

  const Expression num_elements_expr = std::accumulate(
      dims.begin(), dims.end(), Expression(1), std::multiplies<Expression>());
  RETURN_IF_FALSE(SymEngine::is_a_Number(num_elements_expr));
  const int num_elements = static_cast<int>(num_elements_expr);

  context->SetContent(node->name(), ExprVec(num_elements, value[0]));

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Fill", FillOpInferFn);

} // namespace feature_opt
} // namespace tensorflow