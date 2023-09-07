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

#include "symbolic_shape_fn_registry.h"
#include "symbolic_shape_fn.h"

namespace tensorflow {
namespace feature_opt {

SymbolicShapeFnRegistry *SymbolicShapeFnRegistry::Global() {
  static SymbolicShapeFnRegistry *global = new SymbolicShapeFnRegistry;
  return global;
}

void SymbolicShapeFnRegistry::Register(const std::string op,
                                       std::unique_ptr<SymbolicShapeFn> fn) {
  op_fn_mapping[op] = std::move(fn);
}

bool SymbolicShapeFnRegistry::Run(std::shared_ptr<SymbolicShapeContext> context,
                                  NodeDef *node) {
  LOG_AND_RETURN_IF_FALSE(OpRegistered(node->op()),
                          "Unsupported op " + node->op());
  return op_fn_mapping.at(node->op())->Infer(context, node);
}

bool RunSymbolicFn(std::shared_ptr<SymbolicShapeContext> context,
                   NodeDef *node) {
  const std::vector<std::vector<int>> output_shapes =
      FetchGrapplerOutputShapes(node);
  if (SymbolicShapeFnRegistry::Global()->Run(context, node)) {
    bool pass = true;
    for (int i = 0; i < output_shapes.size(); ++i) {
      const std::vector<int> &output_shape = output_shapes[i];
      ExprVec expr_shape = context->GetShape(FormTensorName(node, i));
      bool pass_i = true;
      for (int j = 0; j < output_shape.size(); ++j) {
        if (output_shape[j] >= 0) {
          if (expr_shape[j] != Expression(output_shape[j])) {
            pass_i = false;
            if (!context->MakeEq(expr_shape[j], Expression(output_shape[j]))) {
              RECOM_VLOG << "MakeEq fail";
              expr_shape[j] = Expression(output_shape[j]);
            }
          }
        }
      }

      if (!pass_i) {
        RECOM_VLOG_WARNING << "Value-check of symbolic shape fail! "
                           << FormTensorName(node, i) << " op: " << node->op();
        RECOM_VLOG_WARNING << "Expect " << IntVecToStr(output_shape)
                           << ", but get "
                           << ExprVecToStr(
                                  context->GetShape(FormTensorName(node, i)));
        context->SetShape(FormTensorName(node, i), expr_shape);
        pass = false;
      }
    }

    return pass;
  } else {
    RECOM_VLOG << "Using default symbolic shape of " << node->name();
    for (int i = 0; i < output_shapes.size(); ++i) {
      const std::vector<int> &output_shape = output_shapes[i];
      ExprVec expr_shape(output_shape.size());
      for (int j = 0; j < output_shape.size(); ++j) {
        expr_shape[j] = output_shape[j] >= 0 ? Expression(output_shape[j])
                                             : context->AddNewSymbol(node);
      }
      context->SetShape(FormTensorName(node, i), expr_shape);
    }
    return false;
  }
}

} // namespace feature_opt
} // namespace tensorflow