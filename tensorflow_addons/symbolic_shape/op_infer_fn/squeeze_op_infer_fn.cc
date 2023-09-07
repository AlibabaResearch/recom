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

#include "squeeze_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"

namespace tensorflow {
namespace feature_opt {

bool SqueezeOpInferFn::InferSqueezeShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  const ExprVec input_shape = context->GetShape(node->input(0));

  // WARNING: squeeze_dims is going to be deprecated
  // TODO: 1. support newer axis argument 2. support empty squeeze_dims/axis
  std::vector<bool> squeeze_flags(input_shape.size(), false);
  auto squeeze_dims = node->attr().at("squeeze_dims").list();
  for (int i = 0; i < squeeze_dims.i_size(); ++i) {
    int squeeze_dim = squeeze_dims.i(i);
    if (squeeze_dim < 0)
      squeeze_dim += input_shape.size();
    squeeze_flags[squeeze_dim] = true;
    if (!context->IsEq(input_shape[squeeze_dim], Expression(1))) {
      if (!context->MakeEq(input_shape[squeeze_dim], Expression(1))) {
        RECOM_VLOG << "MakeEq fail to let input dim size be 1";
      }
    }
  }

  // no repeated dims
  ExprVec output_shape(input_shape.size() - squeeze_dims.i_size());
  for (int i = 0, j = 0; i < input_shape.size(); ++i) {
    if (!squeeze_flags[i]) {
      output_shape[j] = input_shape[i];
      ++j;
    }
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

bool SqueezeOpInferFn::InferSqueezeContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  context->SetContent(node->name(), context->GetContent(node->input(0)));
  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Squeeze", SqueezeOpInferFn);

} // namespace feature_opt
} // namespace tensorflow