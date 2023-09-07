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

#include "shape_construct_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <vector>
#include <regex>

namespace tensorflow {
namespace feature_opt {

bool ShapeConstructInferFn::InferShapeConstructShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  const int num_output = node->attr().at("exprs").list().s_size();
  context->SetShape(node->name(), {Expression(num_output)});

  return true;
}

bool ShapeConstructInferFn::InferShapeConstructContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  auto &symbol_list = *(*node->mutable_attr())["symbols"].mutable_list();
  auto &expr_list = *(*node->mutable_attr())["exprs"].mutable_list();
  const auto &indice_list = node->attr().at("indices").list();

  auto cvt_list_to_expr_vec = [&](const AttrValue_ListValue &list) {
    ExprVec expr_vec(list.s_size());
    for (int i = 0; i < list.s_size(); ++i) {
      std::string s = list.s(i);
      s = std::regex_replace(s, std::regex("x"), "a");
      expr_vec[i] = Expression(s);
    }
    return expr_vec;
  };

  const ExprVec &symbols = cvt_list_to_expr_vec(symbol_list);
  ExprVec exprs = cvt_list_to_expr_vec(expr_list);

  for (int i = 0; i < symbols.size(); ++i) {
    RETURN_IF_FALSE(context->ContentKnown(node->input(i)));
    const ExprVec input = context->GetContent(node->input(i));
    const int indice = indice_list.i(i);
    const Expression substitute = input[indice];
    const Expression orig = symbols[i];
    for (Expression &expr : exprs) {
      expr = expr.subs({{orig, substitute}});
    }
    symbol_list.set_s(i, SymEngine::str(substitute));
  }

  for (int i = 0; i < exprs.size(); ++i) {
    expr_list.set_s(i, SymEngine::str(exprs[i]));
  }
  context->SetContent(node->name(), exprs);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("Addons>ShapeConstruct", ShapeConstructInferFn);

} // namespace feature_opt
} // namespace tensorflow