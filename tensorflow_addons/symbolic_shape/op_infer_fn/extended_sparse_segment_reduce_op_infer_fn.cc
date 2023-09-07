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

#include "extended_sparse_segment_reduce_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <functional>
#include <numeric>
#include <vector>

namespace tensorflow {
namespace feature_opt {

bool ExtendedSparseSegmentReduceOpInferFn::
    InferExtendedSparseSegmentReduceShape(
        std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  Expression num_output;
  bool output_determined = false;
  if (context->ContentKnown(node->input(3))) {
    const ExprVec dense_shape = context->GetContent(node->input(3));
    if (dense_shape.back() == Expression(1)) {
      RETURN_IF_FALSE(context->ShapeKnown(node->input(2)));
      num_output = context->GetShape(node->input(2))[0];
      output_determined = true;
    }
  }

  if (!output_determined)
    num_output = context->AddNewSymbol(node);

  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  const ExprVec weight_shape = context->GetShape(node->input(0));
  const Expression embedd_dim = weight_shape.back();

  RETURN_IF_FALSE(context->ShapeKnown(node->input(4)));
  const ExprVec dense_prefix_shape = context->GetShape(node->input(4));
  const Expression output_rank = dense_prefix_shape[0];

  context->SetShape(node->name(), {num_output, output_rank});
  context->SetShape(FormTensorName(node, 1), {num_output, embedd_dim});

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Addons>ExtendedSparseSegmentMean",
                           ExtendedSparseSegmentReduceOpInferFn);

REGISTER_SYMBOLIC_SHAPE_FN("Addons>ExtendedSparseSegmentSum",
                           ExtendedSparseSegmentReduceOpInferFn);

} // namespace feature_opt
} // namespace tensorflow