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

#include "strided_slice_op_infer_fn.h"
#include "tensorflow_addons/symbolic_shape/shape_infer_utils.h"
#include "tensorflow_addons/symbolic_shape/symbolic_shape_fn.h"
#include <symengine/number.h>

namespace tensorflow {
namespace feature_opt {

bool StridedSliceOpInferFn::InferStridedSliceShape(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));

  auto get_mask = [&](const std::string &key) -> int {
    if (node->attr().contains(key)) {
      return node->attr().at(key).i();
    }
    return 0;
  };

  int ellipsis_mask = get_mask("ellipsis_mask");
  int begin_mask = get_mask("begin_mask");
  int end_mask = get_mask("end_mask");
  int new_axis_mask = get_mask("new_axis_mask");
  int shrink_axis_mask = get_mask("shrink_axis_mask");

  const Expression begin_count_expr = context->GetShape(node->input(1))[0];
  int begin_count;
  EXPR_TO_INT(begin_count, begin_count_expr);

  int num_add_axis = 0;
  for (int i = 0; i < begin_count; ++i) {
    if (!((1 << i) & ellipsis_mask) && ((1 << i) & new_axis_mask)) {
      num_add_axis++;
    }
  }

  const ExprVec input_shape = context->GetShape(node->input(0));
  const int input_dims = input_shape.size();
  const int effective_dims = input_dims + num_add_axis;

  int effective_ellipsis_mask = 0, effective_new_axis_mask = 0;
  int ellipsis_start_idx = effective_dims, expanded_ellipsis = 0;
  for (int i = 0; i < effective_dims;) {
    if ((1 << i) & ellipsis_mask) {
      ellipsis_start_idx = i;
      int ellipsis_end_idx = std::max(
          i + 1, std::min(i + 1 + num_add_axis + input_dims - begin_count,
                          effective_dims));
      expanded_ellipsis = ellipsis_end_idx - ellipsis_start_idx - 1;

      for (; i < ellipsis_end_idx; ++i) {
        effective_ellipsis_mask |= (1 << i);
      }
      continue;
    }

    if ((1 << (i - expanded_ellipsis)) & new_axis_mask) {
      effective_new_axis_mask |= (1 << i);
    }
    ++i;
  }

#define RETRIEVE_IF_KNOWN(param, tensor_name)                                  \
  bool param##_known = context->ContentKnown(tensor_name);                     \
  ExprVec param;                                                               \
  if (param##_known) {                                                         \
    param = context->GetContent(tensor_name);                                  \
  }

  RETRIEVE_IF_KNOWN(begin, node->input(1));
  RETRIEVE_IF_KNOWN(end, node->input(2));
  RETRIEVE_IF_KNOWN(strides, node->input(3));

#undef RETRIEVE_IF_KNOWN

  ExprVec output_shape;
  int added_ellipsis = 0, added_axises = 0;
  for (int i = 0; i < effective_dims; ++i) {
    if ((1 << i) & effective_ellipsis_mask) {
      added_ellipsis = std::max(0, i - ellipsis_start_idx);
      output_shape.push_back(input_shape[i - added_axises]);
    } else if ((1 << i) & effective_new_axis_mask) {
      output_shape.push_back(1);
      added_axises++;
    } else if (i >= begin_count + expanded_ellipsis) {
      output_shape.push_back(input_shape[i - added_axises]);
    } else {
      const int orig_idx = i - added_ellipsis;
      if (!(shrink_axis_mask & (1 << orig_idx))) {

        auto handle_negative = [&](const Expression &idx,
                                   const Expression &size) {
          if (idx == Expression(0))
            return idx;
          return idx + MinWithZero(idx) * size / idx;
        };

#define GET_IDX(param, default_value)                                          \
  Expression param##_idx;                                                      \
  if ((param##_mask) & (1 << orig_idx)) {                                      \
    param##_idx = (default_value);                                             \
  } else if (param##_known) {                                                  \
    param##_idx =                                                              \
        handle_negative((param)[orig_idx], input_shape[i - added_axises]);     \
  } else {                                                                     \
    add_new_symbol_flag = true;                                                \
  }

        bool add_new_symbol_flag = false;
        GET_IDX(begin, 0);
        GET_IDX(end, input_shape[i - added_axises]);
        Expression stride;
        strides_known ? strides[orig_idx] : context->AddNewSymbol(node);

#undef GET_IDX

        if (!add_new_symbol_flag && (end_idx - begin_idx) == Expression(1)) {
          output_shape.push_back(1);
        } else if (add_new_symbol_flag || !strides_known) {
          output_shape.push_back(context->AddNewSymbol(node));
        } else {
          Expression num =
              MaxWithZero(Ceiling((end_idx - begin_idx) / strides[orig_idx]));
          output_shape.push_back(num);
        }
      }
    }
  }

  context->SetShape(node->name(), output_shape);

  return true;
}

bool StridedSliceOpInferFn::InferStridedSliceContent(
    std::shared_ptr<SymbolicShapeContext> context, NodeDef *node) {
  // RETURN_IF_FALSE(InputContentKnown(node));
  // RETURN_IF_FALSE(ContentStatic(node->input(1)));
  // RETURN_IF_FALSE(ContentStatic(node->input(2)));
  // RETURN_IF_FALSE(ContentStatic(node->input(3)));

  RETURN_IF_FALSE(context->ShapeKnown(node->name()));
  ExprVec output_shape = context->GetShape(node->name());
  Expression output_size_expr =
      std::accumulate(output_shape.begin(), output_shape.end(), Expression(1),
                      std::multiplies<Expression>());

  RETURN_IF_FALSE(SymEngine::is_a_Number(output_size_expr));
  int output_size = static_cast<int>(output_size_expr);

  // TODO: handle special case whose output is constant
  // and does not depend on the input tensor
  if (output_size == 0) {
    context->SetContent(node->name(), ExprVec());
    RECOM_VLOG << "StridedSlice output []";
    return true;
  }

  RETURN_IF_FALSE(context->ShapeKnown(node->input(0)));
  RETURN_IF_FALSE(context->ShapeKnown(node->input(1)));
  RETURN_IF_FALSE(context->ContentKnown(node->input(0)));

  auto get_mask = [&](const std::string &key) -> int {
    if (node->attr().contains(key)) {
      return node->attr().at(key).i();
    }
    return 0;
  };

  int ellipsis_mask = get_mask("ellipsis_mask");
  int begin_mask = get_mask("begin_mask");
  int end_mask = get_mask("end_mask");
  int new_axis_mask = get_mask("new_axis_mask");
  int shrink_axis_mask = get_mask("shrink_axis_mask");

  const Expression begin_count_expr = context->GetShape(node->input(1))[0];
  int begin_count;
  EXPR_TO_INT(begin_count, begin_count_expr);

  int num_add_axis = 0;
  for (int i = 0; i < begin_count; ++i) {
    if (!((1 << i) & ellipsis_mask) && ((1 << i) & new_axis_mask)) {
      num_add_axis++;
    }
  }

  const ExprVec input_shape = context->GetShape(node->input(0));
  const int input_dims = input_shape.size();
  const int effective_dims = input_dims + num_add_axis;

  int effective_ellipsis_mask = 0, effective_new_axis_mask = 0;
  int ellipsis_start_idx = effective_dims, expanded_ellipsis = 0;
  for (int i = 0; i < effective_dims;) {
    if ((1 << i) & ellipsis_mask) {
      ellipsis_start_idx = i;
      int ellipsis_end_idx = std::max(
          i + 1, std::min(i + 1 + num_add_axis + input_dims - begin_count,
                          effective_dims));
      expanded_ellipsis = ellipsis_end_idx - ellipsis_start_idx - 1;

      for (; i < ellipsis_end_idx; ++i) {
        effective_ellipsis_mask |= (1 << i);
      }
      continue;
    }

    if ((1 << (i - expanded_ellipsis)) & new_axis_mask) {
      effective_new_axis_mask |= (1 << i);
    }
    ++i;
  }

#define RETRIEVE_IF_KNOWN(param, tensor_name)                                  \
  bool param##_known = context->ContentKnown(tensor_name);                     \
  ExprVec param;                                                               \
  if (param##_known) {                                                         \
    param = context->GetContent(tensor_name);                                  \
  }

  RETRIEVE_IF_KNOWN(begin, node->input(1));
  RETRIEVE_IF_KNOWN(end, node->input(2));
  RETRIEVE_IF_KNOWN(strides, node->input(3));

#undef RETRIEVE_IF_KNOWN

  std::vector<int> effective_input_shape(effective_dims);
  std::vector<int> numeric_begin(effective_dims);
  std::vector<int> numeric_end(effective_dims);
  std::vector<int> numeric_strides(effective_dims);
  int added_ellipsis = 0, added_axises = 0;
  for (int i = 0; i < effective_dims; ++i) {
    if ((1 << i) & effective_ellipsis_mask) {
      added_ellipsis = std::max(0, i - ellipsis_start_idx);

      int end_idx;
      EXPR_TO_INT(end_idx, input_shape[i - added_axises]);
      numeric_begin[i] = 0;
      numeric_end[i] = end_idx;
      numeric_strides[i] = 1;
      effective_input_shape[i] = end_idx;
    } else if ((1 << i) & effective_new_axis_mask) {
      numeric_begin[i] = 0;
      numeric_end[i] = 1;
      numeric_strides[i] = 1;
      effective_input_shape[i] = 1;

      added_axises++;
    } else if (i >= begin_count + expanded_ellipsis) {
      int end_idx;
      EXPR_TO_INT(end_idx, input_shape[i - added_axises]);
      numeric_begin[i] = 0;
      numeric_end[i] = end_idx;
      numeric_strides[i] = 1;
      effective_input_shape[i] = end_idx;
    } else {
      const int orig_idx = i - added_ellipsis;
      EXPR_TO_INT(effective_input_shape[i], input_shape[i - added_axises]);

#define RETRIEVE_IF_EXIST(res, param)                                          \
  RETURN_IF_FALSE(param##_known);                                              \
  int res;                                                                     \
  EXPR_TO_INT(res, param[orig_idx]);

#define GET_IDX(param, default_value)                                          \
  int param##_idx;                                                             \
  if ((param##_mask) & (1 << orig_idx)) {                                      \
    param##_idx = default_value;                                               \
  } else {                                                                     \
    RETRIEVE_IF_EXIST(param##_tmp, param);                                     \
    param##_idx = param##_tmp >= 0 ? param##_tmp                               \
                                   : param##_tmp + effective_input_shape[i];   \
  }                                                                            \
  numeric_##param[i] = param##_idx;

      // TODO: can be more aggressive.
      // In some special cases, we can infer the content even if the
      // input shape is not entirely static
      GET_IDX(begin, 0);
      if (shrink_axis_mask & (1 << orig_idx)) {
        numeric_end[i] = begin_idx + 1;
        numeric_strides[i] = 1;
      } else {
        GET_IDX(end, effective_input_shape[i]);
        if (begin_idx + 1 == end_idx) {
          numeric_strides[i] = 1;
        } else {
          RETRIEVE_IF_EXIST(stride, strides);
          numeric_strides[i] = stride;
        }
      }

#undef GET_IDX
#undef RETRIEVE
    }
  }

  const std::vector<int> offset_units =
      ComputeOffsetUnits(effective_input_shape, 1);

  ExprVec input = context->GetContent(node->input(0));
  ExprVec output(output_size);
  auto itr = output.begin();
  std::function<bool(int, int)> strided_slice = [&](int i, int offset) -> bool {
    int begin_idx = numeric_begin[i];
    int end_idx = numeric_end[i];
    int stride = numeric_strides[i];
    if (i + 1 == effective_dims) {
      for (int idx = begin_idx; idx < end_idx; idx += stride) {
        *(itr++) = input[offset + idx];
      }
    } else {
      for (int idx = begin_idx; idx < end_idx; idx += stride) {
        RETURN_IF_FALSE(strided_slice(i + 1, offset + idx * offset_units[i]));
      }
    }
    return true;
  };
  RETURN_IF_FALSE(strided_slice(0, 0));

  context->SetContent(node->name(), output);

  return true;
}

REGISTER_SYMBOLIC_SHAPE_FN("StridedSlice", StridedSliceOpInferFn);

} // namespace feature_opt
} // namespace tensorflow