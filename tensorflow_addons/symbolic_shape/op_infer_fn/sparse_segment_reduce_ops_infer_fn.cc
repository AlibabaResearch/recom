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

#include "sparse_segment_reduce_ops_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool SparseSegmentReduceOpInferFn::InferSparseSegmentReduceShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(2)));
  ExprVec data_shape = context->GetShape(node->input(0));
  const ExprVec indice_shape = context->GetShape(node->input(1));
  const ExprVec segment_id_shape = context->GetShape(node->input(2));
  assert(data_shape.size() > 0);
  assert(indice_shape.size() == 1 && segment_id_shape.size() == 1);
  assert(indice_shape[0] == segment_id_shape[0]);

  data_shape[0] = context->AddNewSymbol(node);
  context->SetShape(node->name(), data_shape);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("SparseSegmentMean", SparseSegmentReduceOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("SparseSegmentSum", SparseSegmentReduceOpInferFn);

bool SparseSegmentReduceWithNumSegmentsOpInferFn::
    InferSparseSegmentReduceWithNumSegmentsShape(
        std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(2)));
  ExprVec data_shape = context->GetShape(node->input(0));
  const ExprVec indice_shape = context->GetShape(node->input(1));
  const ExprVec segment_id_shape = context->GetShape(node->input(2));
  assert(data_shape.size() > 0);
  assert(indice_shape.size() == 1 && segment_id_shape.size() == 1);
  assert(indice_shape[0] == segment_id_shape[0]);

  RETURN_IF_FALSE(context->ContentKnown(node->input(3)));
  const ExprVec num_segments = context->GetContent(node->input(3));
  assert(num_segments.size() == 1);

  data_shape[0] = num_segments[0];
  context->SetShape(node->name(), data_shape);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("SparseSegmentMeanWithNumSegments",
                           SparseSegmentReduceWithNumSegmentsOpInferFn);
REGISTER_SYMBOLIC_SHAPE_FN("SparseSegmentSumWithNumSegments",
                           SparseSegmentReduceWithNumSegmentsOpInferFn);

} // namespace feature_opt
} // namespace tensorflow