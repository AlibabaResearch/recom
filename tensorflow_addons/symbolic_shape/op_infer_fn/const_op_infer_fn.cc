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

#include "const_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <symengine/expression.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/types.pb.h>

namespace tensorflow {
namespace feature_opt {

bool ConstOpInferFn::InferConstShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  Tensor tensor;
  LOG_AND_RETURN_IF_FALSE(tensor.FromProto(node->attr().at("value").tensor()),
                          "Convert TensorProto of Const op to Tensor fail");

  const TensorShape &tensor_shape = tensor.shape();
  if (tensor_shape.dims() > 0) {
    ExprVec expr_shape(tensor_shape.dims());
    for (int i = 0; i < tensor_shape.dims(); ++i) {
      expr_shape[i] = Expression(tensor_shape.dim_size(i));
    }
    context->SetShape(node->name(), expr_shape);
  } else { // scalar
    // context->SetShape(node->name(), {Expression(1)});
    context->SetShape(node->name(), {});
  }

  return true;
}

bool ConstOpInferFn::InferConstContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  Tensor tensor;
  LOG_AND_RETURN_IF_FALSE(tensor.FromProto(node->attr().at("value").tensor()),
                          "Convert TensorProto of Const op to Tensor fail");

  ExprVec content(tensor.NumElements());
  switch (tensor.dtype()) {
  case DT_INT32:
    CpyDataToExpr<int>(tensor, content);
    break;
  case DT_INT64:
    CpyDataToExpr<int64>(tensor, content);
    break;
  case DT_FLOAT:
    CpyDataToExpr<float>(tensor, content);
    break;
  case DT_DOUBLE:
    CpyDataToExpr<double>(tensor, content);
    break;
  case DT_STRING:
    // RECOM_VLOG << "Assign new symbol for Const op with string type during "
    //               "symbolic shape inference";
    // for (Expression &expr : content)
    //   expr = context->AddNewSymbol(node);
    // break;
  default:
    // TODO: handle more type
    RECOM_VLOG_WARNING << "Unsupported type of Const "
                       << DataTypeString(tensor.dtype());
    context->SetContent(node->name(), content); // do not use it
    RETURN_FALSE;
  }

  context->SetContent(node->name(), content);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Const", ConstOpInferFn);

} // namespace feature_opt
} // namespace tensorflow