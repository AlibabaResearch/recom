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

#include "reshape_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"

namespace tensorflow {
namespace feature_opt {

bool ReshapeOpInferFn::InferReshapeShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));

  ExprVec input_shape = context->GetShape(node->input(0));
  Expression input_size =
      std::accumulate(input_shape.begin(), input_shape.end(), Expression(1),
                      std::multiplies<Expression>());

  ExprVec shape_shape = context->GetShape(node->input(1));
  Expression rank_expr =
      std::accumulate(shape_shape.begin(), shape_shape.end(), Expression(1),
                      std::multiplies<Expression>());
  int rank;
  EXPR_TO_INT(rank, rank_expr);

  ExprVec output_shape(rank);
  if (context->ContentKnown(node->input(1))) {
    ExprVec shape_arg = context->GetContent(node->input(1));
    assert(rank == shape_arg.size());

    int infer_pos = -1;
    Expression known_prod(1);
    for (int i = 0; i < rank; ++i) {
      Expression n = shape_arg[i];
      if (SymEngine::is_a_Number(n) && static_cast<int>(n) < 0) {
        LOG_AND_RETURN_IF_FALSE(infer_pos == -1,
                                "Unhandled condition: multiple -1");
        infer_pos = i;
      } else {
        output_shape[i] = n;
        known_prod *= n;
      }
    }

    if (infer_pos >= 0)
      output_shape[infer_pos] = input_size / known_prod;
  } else {
    RECOM_VLOG << "Reshape with shape input unknown";

    Expression accum(1);
    for (int i = 0; i < rank - 1; ++i) {
      output_shape[i] = context->AddNewSymbol(node);
      accum *= output_shape[i];
    }
    output_shape.back() = input_size / accum;
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

bool ReshapeOpInferFn::InferReshapeContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  context->SetContent(node->name(), context->GetContent(node->input(0)));
  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Reshape", ReshapeOpInferFn);

} // namespace feature_opt
} // namespace tensorflow