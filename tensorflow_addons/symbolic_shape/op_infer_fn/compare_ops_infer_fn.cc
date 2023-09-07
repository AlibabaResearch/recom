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

#include "compare_ops_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"

namespace tensorflow {
namespace feature_opt {

bool CompareOpInferFn::InferCompareShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));

  ExprVec output_shape;
  RETURN_IF_FALSE(GetBroadCastShape(context, context->GetShape(node->input(0)),
                                    context->GetShape(node->input(1)),
                                    output_shape));

  context->SetShape(node->name(), output_shape);
  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("GreaterEqual", CompareOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("Greater", CompareOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("LessEqual", CompareOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("Less", CompareOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("NotEqual", CompareOpInferFn);

} // namespace feature_opt
} // namespace tensorflow