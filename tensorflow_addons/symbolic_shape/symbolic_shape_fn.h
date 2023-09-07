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
#include <functional>
#include <numeric>
#include <string>
#include <symengine/expression.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/number.h>
#include <vector>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

using SymEngine::Expression;
using ExprVec = std::vector<Expression>;

#define EXPR_TO_INT(res, expr)                                                 \
  RETURN_IF_FALSE(SymEngine::is_a_Number(expr));                               \
  (res) = static_cast<int>(expr);

std::string ExprVecToStr(const ExprVec &expr_vec);

std::string IntVecToStr(const std::vector<int> &vec);

class SymbolicShapeContext {
  constexpr static int MAX_ATTEMP = 1 << 8;
  constexpr static int MIN_ATTEMP = -MAX_ATTEMP;

  struct Symbol {
    Expression expr;
    NodeDef *node; // node that generates the symbol

    int idx;
    int parent_idx;

    Symbol() = default;
    Symbol(int idx, NodeDef *node)
        : expr("x" + std::to_string(idx)), node(node), idx(idx),
          parent_idx(-1) {}
  };

  std::vector<Symbol> symbols;

  HashMapT<std::string, ExprVec> symbolic_shape_mapping;
  HashMapT<std::string, ExprVec> symbolic_content_mapping;
  HashMapT<std::string, NodeDef *> tensor_shape_node_mapping;

private:
  int FindParentIdx(int idx);

  bool IsNumeric(int idx) { return SymEngine::is_a_Number(symbols[idx].expr); }

  bool UnionSymbols(int a, int b);

  bool SetSymbol(int idx, int value);

  std::set<int> RetrieveSymbolIdxSet(const Expression &expr);

  Expression SubsWithParent(Expression expr);

  void SubsAll();

  void SubsAll(const Expression &a, const Expression &b);

public:
  SymbolicShapeContext() = default;

  Expression AddNewSymbol(NodeDef *node);

  bool IsStatic(const ExprVec &exprs);

  bool IsSymbol(const Expression &expr);

  bool IsEq(const Expression &a, const Expression &b);

  bool IsEq(const ExprVec &a, const ExprVec &b);

  bool MakeEq(const Expression &a, const Expression &b);

  bool ShapeKnown(const std::string &tensor_name);

  bool ShapeStatic(const std::string &tensor_name);

  bool ShapeStatic(const std::string &tensor_name, std::vector<int> &shape);

  ExprVec GetShape(const std::string &tensor_name);

  void SetShape(const std::string &tensor_name, const ExprVec &shape);

  bool ContentKnown(const std::string &tensor_name);

  bool InputContentKnown(NodeDef *node);

  bool ContentStatic(const std::string &tensor_name);

  bool ContentStatic(const std::string &tensor_name, std::vector<int> &content);

  ExprVec GetContent(const std::string &tensor_name);

  void SetContent(const std::string &tensor_name, const ExprVec &content);

  bool HasShapeNode(const std::string &tensor_name);

  NodeDef *GetShapeNode(const std::string &tensor_name);

  void RecordTensorShapeNode(const std::string &tensor_name, NodeDef *node);

  std::vector<std::pair<Expression, NodeDef *>>
  RetrieveSymbolExprGenNodePairs(const Expression &expr);

  std::vector<std::pair<Expression, NodeDef *>>
  FindEqSymbolExprGenNodePairs(const Expression &symbol_expr);
};

class SymbolicShapeFn {
public:
  SymbolicShapeFn() = default;
  virtual ~SymbolicShapeFn() = default;

  virtual bool Infer(std::shared_ptr<SymbolicShapeContext> context,
                     NodeDef *node) = 0;
};

} // namespace feature_opt
} // namespace tensorflow