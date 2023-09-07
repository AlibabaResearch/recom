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

#include "sparse_fill_empty_rows_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"

namespace tensorflow {
namespace feature_opt {

bool SparseFillEmptyRowsOpInferFn::ConstrainInput(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  const ExprVec indice_shape = context->GetShape(node->input(0));
  const ExprVec value_shape = context->GetShape(node->input(1));

  assert(indice_shape.size() == 2 && value_shape.size() == 1);

  bool ret = true;
  if (!context->IsEq(indice_shape[0], value_shape[0])) {
    if (!context->MakeEq(indice_shape[0], value_shape[0])) {
      RECOM_VLOG
          << "Cannot constrain SparseFillEmptyRows inputs (indice and value)";
      ret = false;
    }
  }

  RETURN_IF_FALSE(context->ShapeKnown(node->input(2)));
  const ExprVec dense_shape_shape = context->GetShape(node->input(2));
  assert(dense_shape_shape.size() == 1);

  RETURN_IF_FALSE(dense_shape_shape[0] == indice_shape[1]);

  return ret;
}

bool SparseFillEmptyRowsOpInferFn::InferSparseFillEmptyRowsShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  const ExprVec indice_shape = context->GetShape(node->input(0));
  context->SetShape(FormTensorName(node, 3), {indice_shape[0]});

  if (context->ContentKnown(node->input(2))) {
    const ExprVec dense_shape = context->GetContent(node->input(2));
    const Expression num_rows = dense_shape[0];
    context->SetShape(FormTensorName(node, 2), {num_rows});

    bool all_one = true;
    for (int i = 1; i < dense_shape.size(); ++i) {
      if (dense_shape[i] != Expression(1)) {
        all_one = false;
        break;
      }
    }

    if (all_one) { // special case
      context->SetShape(FormTensorName(node, 0), {num_rows, indice_shape[1]});
      context->SetShape(FormTensorName(node, 1), {num_rows});
    } else {
      const Expression n = context->AddNewSymbol(node);
      context->SetShape(FormTensorName(node, 0), {n, indice_shape[1]});
      context->SetShape(FormTensorName(node, 1), {n});
    }
  } else {
    RECOM_VLOG_WARNING << "The content of dense_shape input of "
                          "SparseFillEmptyRows is not known";

    const Expression num_rows = context->AddNewSymbol(node);
    context->SetShape(FormTensorName(node, 2), {num_rows});

    const Expression n = context->AddNewSymbol(node);
    context->SetShape(FormTensorName(node, 0), {n, indice_shape[1]});
    context->SetShape(FormTensorName(node, 1), {n});
  }

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("SparseFillEmptyRows", SparseFillEmptyRowsOpInferFn);

} // namespace feature_opt
} // namespace tensorflow