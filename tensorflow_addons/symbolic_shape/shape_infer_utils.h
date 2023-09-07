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

#include "symbolic_shape_fn.h"
#include "symbolic_shape_fn_registry.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

bool GetBroadCastShape(std::shared_ptr<SymbolicShapeContext> context,
                       const ExprVec &a_shape, const ExprVec &b_shape,
                       ExprVec &output_shape);

int UnsafeMod(const Expression &expr, const Expression &symbol, unsigned int M);

Expression Ceiling(const Expression &expr);

Expression Floor(const Expression &expr);

Expression Min(const Expression &a, const Expression &b);

Expression Max(const Expression &a, const Expression &b);

Expression MinWithZero(const Expression &expr);

Expression MaxWithZero(const Expression &expr);

template <typename T>
std::vector<T> ComputeOffsetUnits(const std::vector<T> &shape, T init) {
  std::vector<T> offset_units(shape.size(), init);
  for (int i = shape.size() - 1; i > 0; --i) {
    offset_units[i - 1] = offset_units[i] * shape[i];
  }
  return offset_units;
}

} // namespace feature_opt
} // namespace tensorflow