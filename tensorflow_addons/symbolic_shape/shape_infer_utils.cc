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

#include "shape_infer_utils.h"
#include <symengine/functions.h>
#include <symengine/number.h>

namespace tensorflow {
namespace feature_opt {

bool GetBroadCastShape(std::shared_ptr<SymbolicShapeContext> context,
                       const ExprVec &a_shape, const ExprVec &b_shape,
                       ExprVec &output_shape) {
  if (a_shape.size() == 0) {
    output_shape = b_shape;
  } else if (b_shape.size() == 0) {
    output_shape = a_shape;
  } else {
    output_shape = ExprVec(std::max(a_shape.size(), b_shape.size()));
    auto output_itr = output_shape.rbegin();
    auto a_itr = a_shape.crbegin();
    auto b_itr = b_shape.crbegin();
    while (a_itr != a_shape.crend() && b_itr != b_shape.crend()) {
      if (context->IsEq(*a_itr, *b_itr)) {
        *output_itr = *a_itr;
      } else {
        if (*a_itr == Expression(1)) {
          *output_itr = *b_itr;
        } else if (*b_itr == Expression(1)) {
          *output_itr = *a_itr;
        } else if (!SymEngine::is_a_Number(*a_itr) &&
                   !SymEngine::is_a_Number(*b_itr)) {
          RECOM_VLOG_WARNING << "Not know how to handle the broadcast "
                                "operation with two different expressions";
          *output_itr = Max(*a_itr, *b_itr);
        } else if (SymEngine::is_a_Number(*a_itr) &&
                   !SymEngine::is_a_Number(*b_itr)) {
          *output_itr = *a_itr;
        } else if (!SymEngine::is_a_Number(*a_itr) &&
                   SymEngine::is_a_Number(*b_itr)) {
          *output_itr = *b_itr;
        } else {
          LOG_AND_RETURN_FALSE("Fail to handle the broadcast!");
        }
      }
      ++a_itr, ++b_itr;
      ++output_itr;
    }

    while (a_itr != a_shape.crend()) {
      *output_itr = *a_itr;
      ++a_itr;
      ++output_itr;
    }

    while (b_itr != b_shape.crend()) {
      *output_itr = *b_itr;
      ++b_itr;
      ++output_itr;
    }
  }

  return true;
}

int UnsafeMod(const Expression &expr, const Expression &symbol,
              unsigned int M) {
  int remainder = -1;
  for (unsigned int i = 0; i < M; ++i) {
    auto subs = expr.subs({{symbol, SymEngine::integer(i)}});
    if (SymEngine::is_a_Number(subs)) {
      int remainder_tmp = static_cast<int>(subs) % M;
      if (remainder == -1) {
        remainder = remainder_tmp;
      } else if (remainder != remainder_tmp) {
        return -1;
      }
    } else {
      return -1;
    }
  }

  return remainder;
}

Expression Ceiling(const Expression &expr) {
  std::string str = SymEngine::str(expr);
  bool is_integer = true;
  for (char ch : str) {
    if (ch == '/' || ch == '.') {
      is_integer = false;
      break;
    }
  }

  if (is_integer) {
    return expr;
  } else {
    return SymEngine::ceiling(expr);
  }
}

Expression Floor(const Expression &expr) {
  std::string str = SymEngine::str(expr);
  bool is_integer = true;
  for (char ch : str) {
    if (ch == '/' || ch == '.') {
      is_integer = false;
      break;
    }
  }

  if (is_integer) {
    return expr;
  } else {
    return SymEngine::floor(expr);
  }
}

Expression Min(const Expression &a, const Expression &b) {
  return SymEngine::min({a, b});
}

Expression Max(const Expression &a, const Expression &b) {
  return SymEngine::max({a, b});
}

Expression MinWithZero(const Expression &expr) {
  std::string str = SymEngine::str(expr);
  bool non_negative = true;
  for (char ch : str) {
    if (ch == '-') {
      non_negative = false;
      break;
    }
  }

  if (non_negative) {
    return Expression(0);
  } else {
    return SymEngine::min({expr, Expression(0)});
  }
}

Expression MaxWithZero(const Expression &expr) {
  std::string str = SymEngine::str(expr);
  bool non_negative = true;
  for (char ch : str) {
    if (ch == '-') {
      non_negative = false;
      break;
    }
  }

  if (non_negative) {
    return expr;
  } else {
    return SymEngine::max({expr, Expression(0)});
  }
}

} // namespace feature_opt
} // namespace tensorflow