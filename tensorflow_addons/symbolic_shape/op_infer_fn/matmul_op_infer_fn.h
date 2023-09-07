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

#pragma once
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn_registry.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

class MatMulOpInferFn : public SymbolicShapeFn {
  bool ConstrainInput(std::shared_ptr<SymbolicShapeContext> context,
                      NodeDef *node);

  bool InferMatMulShape(std::shared_ptr<SymbolicShapeContext> context,
                        NodeDef *node);

  bool InferMatMulContent(std::shared_ptr<SymbolicShapeContext> context,
                          NodeDef *node);

public:
  MatMulOpInferFn() = default;

  bool Infer(std::shared_ptr<SymbolicShapeContext> context,
             NodeDef *node) override {
    if (!ConstrainInput(context, node)) {
      RECOM_VLOG << "Fail to constrain input";
    }

    bool shape_flag = true;
    if (!InferMatMulShape(context, node)) {
      RECOM_VLOG << "Fail to infer MatMul shape";
      shape_flag = false;
    }
    if (!InferMatMulContent(context, node)) {
      RECOM_VLOG << "Fail to infer MatMul content";
    }
    return shape_flag;
  }
};

} // namespace feature_opt
} // namespace tensorflow