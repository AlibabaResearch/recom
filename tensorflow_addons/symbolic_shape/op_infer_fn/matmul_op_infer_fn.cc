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

#include "matmul_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <functional>
#include <numeric>
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool MatMulOpInferFn::ConstrainInput(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  ExprVec a_shape = context->GetShape(node->input(0));
  ExprVec b_shape = context->GetShape(node->input(1));
  assert(a_shape.size() == 2 && b_shape.size() == 2);

  if (node->attr().at("transpose_a").b())
    std::swap(a_shape[0], a_shape[1]);
  if (node->attr().at("transpose_b").b())
    std::swap(b_shape[0], b_shape[1]);

  if (!context->IsEq(a_shape[1], b_shape[0])) {
    RETURN_IF_FALSE(context->MakeEq(a_shape[1], b_shape[0]));
  }

  return true;
}

bool MatMulOpInferFn::InferMatMulShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  ExprVec a_shape = context->GetShape(node->input(0));
  ExprVec b_shape = context->GetShape(node->input(1));
  assert(a_shape.size() == 2 && b_shape.size() == 2);

  if (node->attr().at("transpose_a").b())
    std::swap(a_shape[0], a_shape[1]);
  if (node->attr().at("transpose_b").b())
    std::swap(b_shape[0], b_shape[1]);

  context->SetShape(node->name(), {a_shape[0], b_shape[1]});

  return true;
}

bool MatMulOpInferFn::InferMatMulContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  // TODO
  return false;
}

REGISTER_SYMBOLIC_SHAPE_FN("MatMul", MatMulOpInferFn);

} // namespace feature_opt
} // namespace tensorflow