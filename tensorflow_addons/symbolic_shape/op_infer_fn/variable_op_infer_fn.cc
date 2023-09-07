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

#include "variable_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool VariableOpInferFn::InferVariableShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  const std::vector<std::vector<int>> output_shapes =
      FetchGrapplerOutputShapes(node);
  for (int i = 0; i < output_shapes.size(); ++i) {
    const std::vector<int> &output_shape = output_shapes[i];
    ExprVec expr_shape(output_shape.size());
    for (int j = 0; j < output_shape.size(); ++j) {
      assert(output_shape[j] >= 0);
      expr_shape[j] = Expression(output_shape[j]);
    }
    context->SetShape(FormTensorName(node, i), expr_shape);
  }

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("VariableV2", VariableOpInferFn);

bool VarHandleOpInferFn::InferVarHandleShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  const TensorShape &shape = node->attr().at("shape").shape();
  ExprVec expr_shape(shape.dims());
  for (int i = 0; i < shape.dims(); ++i) {
    expr_shape[i] = shape.dim_size(i);
  }
  context->SetShape(node->name(), expr_shape);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("VarHandleOp", VarHandleOpInferFn);

} // namespace feature_opt
} // namespace tensorflow