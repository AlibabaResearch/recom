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

#include "sparse_reshape_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <symengine/functions.h>

namespace tensorflow {
namespace feature_opt {

bool SparseReshapeOpInferFn::InferSparseReshapeShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(2)));
  ExprVec new_shape_shape = context->GetShape(node->input(2));
  assert(new_shape_shape.size() == 1);
  context->SetShape(FormTensorName(node, 1), new_shape_shape);

  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  ExprVec indice_shape = context->GetShape(node->input(0));
  RETURN_IF_FALSE(indice_shape.size() == 2);

  context->SetShape(node->name(),
                    {indice_shape[0], Expression(new_shape_shape[0])});

  return true;
}

bool SparseReshapeOpInferFn::InferSparseReshapeContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(2)));
  const ExprVec new_shape = context->GetContent(node->input(2));
  context->SetContent(FormTensorName(node, 1), new_shape);

  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  RETURN_IF_FALSE(context->ContentKnown(node->input(1)));
  const ExprVec input_indice = context->GetContent(node->input(0));
  const ExprVec orig_shape = context->GetContent(node->input(1));

  const ExprVec input_offset_unit =
      ComputeOffsetUnits(orig_shape, Expression(1));
  const ExprVec output_offset_unit =
      ComputeOffsetUnits(new_shape, Expression(1));

  const int nnz = input_indice.size() / orig_shape.size();
  ExprVec output_indice(nnz * new_shape.size());
  auto input_itr = input_indice.cbegin();
  auto output_itr = output_indice.begin();
  for (int i = 0; i < nnz; ++i) {
    Expression idx(0);
    for (int j = 0; j < orig_shape.size(); ++j) {
      idx += *input_itr * input_offset_unit[j];
      ++input_itr;
    }

    for (int j = 0; j < new_shape.size(); ++j) {
      *output_itr = Floor(idx / output_offset_unit[j]);
      idx -= *output_itr * output_offset_unit[j];
      ++output_itr;
    }
  }

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("SparseReshape", SparseReshapeOpInferFn);

} // namespace feature_opt
} // namespace tensorflow