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

#include "segment_reduce_ops_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool SegmentReduceOpInferFn::InferSegmentReduceShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  const ExprVec data_shape = context->GetShape(node->input(0));
  const ExprVec segment_id_shape = context->GetShape(node->input(1));
  assert(data_shape.size() > 0 && segment_id_shape.size() == 1);
  assert(data_shape[0] == segment_id_shape[0]);

  ExprVec output_shape = data_shape;
  if (context->ContentKnown(node->input(1))) {
    const ExprVec segment_id = context->GetContent(node->input(1));
    output_shape[0] = segment_id.back() + 1;
  } else {
    output_shape[0] = context->AddNewSymbol(node);
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

bool SegmentReduceOpInferFn::InferSegmentReduceContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  // TODO
  return false;
}

REGISTER_SYMBOLIC_SHAPE_FN("SegmentMax", SegmentReduceOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("SegmentMin", SegmentReduceOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("SegmentMean", SegmentReduceOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("SegmentSum", SegmentReduceOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("SegmentProd", SegmentReduceOpInferFn);

} // namespace feature_opt
} // namespace tensorflow