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

#include "simple_ops_infer_fn.h"
#include <tensorflow/core/framework/types.h>

namespace tensorflow {
namespace feature_opt {

bool IdentityOpInferFn::InferIdentityShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  context->SetShape(node->name(), context->GetShape(node->input(0)));
  return true;
}

bool IdentityOpInferFn::InferIdentityContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->InputContentKnown(node));

  context->SetContent(node->name(), context->GetContent(node->input(0)));
  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Identity", IdentityOpInferFn);

bool CastOpInferFn::InferCastShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  context->SetShape(node->name(), context->GetShape(node->input(0)));
  return true;
}

bool CastOpInferFn::InferCastContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->InputContentKnown(node));
  auto srcT = node->attr().at("SrcT").type();
  auto dstT = node->attr().at("DstT").type();

  // TODO: handle various type casting
  if (!(DataTypeIsInteger(srcT) && DataTypeIsInteger(dstT))) {
    RECOM_VLOG_WARNING << "cannot guareent the correctness of Cast op from "
                       << DataTypeString(srcT) << " to "
                       << DataTypeString(dstT);
  }

  context->SetContent(node->name(), context->GetContent(node->input(0)));
  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Cast", CastOpInferFn);

} // namespace feature_opt
} // namespace tensorflow