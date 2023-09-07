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

#include "gather_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <functional>
#include <numeric>

namespace tensorflow {
namespace feature_opt {

bool GatherOpInferFn::InferGatherShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  // TODO: support batch-size attribute
  LOG_AND_RETURN_IF_FALSE(!node->attr().count("batch_dims") ||
                              (node->attr().at("batch_dims").i() == 0),
                          "Currently not support GatherV2 with batch_dims > 0");
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));

  ExprVec param_shape = context->GetShape(node->input(0));
  ExprVec indice_shape = context->GetShape(node->input(1));
  ExprVec output_shape(param_shape.size() + indice_shape.size() - 1);
  if (context->ContentStatic(node->input(2))) {
    int axis = static_cast<int>(context->GetContent(node->input(2))[0]);
    if (axis < 0)
      axis += param_shape.size();

    for (int i = 0; i < axis; ++i) {
      output_shape[i] = param_shape[i];
    }

    for (int i = 0; i < indice_shape.size(); ++i) {
      output_shape[axis + i] = indice_shape[i];
    }

    for (int i = axis + 1; i < param_shape.size(); ++i) {
      output_shape[indice_shape.size() + i - 1] = param_shape[i];
    }
  } else {
    RECOM_VLOG << "Gather with a dynamic axis";
    for (Expression &expr : output_shape) {
      expr = context->AddNewSymbol(node);
    }
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

bool GatherOpInferFn::InferGatherContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->InputContentKnown(node));
  LOG_AND_RETURN_IF_FALSE(!node->attr().count("batch_dims") ||
                              (node->attr().at("batch_dims").i() == 0),
                          "Currently not support GatherV2 with batch_dims > 0");

  ExprVec params = context->GetContent(node->input(0));

  std::vector<int> param_shape;
  RETURN_IF_FALSE(context->ShapeStatic(node->input(0), param_shape));

  std::vector<int> indices;
  RETURN_IF_FALSE(context->ContentStatic(node->input(1), indices));

  RETURN_IF_FALSE(context->ContentStatic(node->input(2)));
  int axis = static_cast<int>(context->GetContent(node->input(2))[0]);
  if (axis < 0)
    axis += param_shape.size();

  const int outer_loop_count =
      std::accumulate(param_shape.begin(), param_shape.begin() + axis, 1,
                      std::multiplies<int>());
  const int inner_loop_count =
      std::accumulate(param_shape.begin() + axis + 1, param_shape.end(), 1,
                      std::multiplies<int>());
  const int size_at_axis = param_shape[axis];
  const int outer_offset_unit = size_at_axis * inner_loop_count;

  ExprVec content(outer_loop_count * indices.size() * inner_loop_count);
  auto itr = content.begin();
  for (int i = 0; i < outer_loop_count; ++i) {
    for (int index : indices) {
      RETURN_IF_FALSE(index >= 0 && index < size_at_axis);
      for (int j = 0; j < inner_loop_count; ++j) {
        *(itr++) =
            params.at(i * outer_offset_unit + index * inner_loop_count + j);
      }
    }
  }

  context->SetContent(node->name(), content);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("GatherV2", GatherOpInferFn);

} // namespace feature_opt
} // namespace tensorflow