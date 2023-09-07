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

#include "scatter_nd_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"

namespace tensorflow {
namespace feature_opt {

bool ScatterNdOpInferFn::InferScatterNdShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(2)));
  context->SetShape(node->name(), context->GetContent(node->input(2)));

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("ScatterNd", ScatterNdOpInferFn);

} // namespace feature_opt
} // namespace tensorflow