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

#include "bucketize_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"

namespace tensorflow {
namespace feature_opt {

bool BucketizeOpInferFn::InferBucketizeShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  context->SetShape(node->name(), context->GetShape(node->input(0)));
  return true;
}

bool BucketizeOpInferFn::InferBucketizeContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));
  LOG_AND_RETURN_FALSE("Currently not support Bucketize content infer");
  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Bucketize", BucketizeOpInferFn);

} // namespace feature_opt
} // namespace tensorflow