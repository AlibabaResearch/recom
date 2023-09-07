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

#include <symengine/expression.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_requires.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/errors.h>
#include <type_traits>
#include <vector>

#include "extended_sparse_to_dense_ops.h"

namespace tensorflow {
namespace feature_opt {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T, typename Tindices, typename Tshape>
class ExtendedSparseToDenseOp : public OpKernel {
private:
  float default_float;

public:
  explicit ExtendedSparseToDenseOp(OpKernelConstruction *c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("default_float", &default_float));
  }

  void Compute(OpKernelContext *c) override {
    if (std::is_same<T, float>::value) {
      functor::ExtendedSparseToDenseFunctor<Device, float, Tindices, Tshape>()(
          c, c->input(0), c->input(1), c->input(2), default_float);
    } else {
      // TODO: support more types
    }
  }
};

namespace functor {

template <typename T, typename Tindices, typename Tshape>
class ExtendedSparseToDenseFunctor<CPUDevice, T, Tindices, Tshape> {
public:
  void operator()(OpKernelContext *c, const Tensor &indices_t,
                  const Tensor &values_t, const Tensor &dense_prefix_t,
                  T default_value) {
    const TensorShape values_shape = values_t.shape();
    TensorShape output_shape;
    for (int i = 0; i < dense_prefix_t.NumElements(); ++i) {
      output_shape.AddDim(dense_prefix_t.flat<Tshape>()(i));
    }
    for (int i = 1; i < values_shape.dims(); ++i) {
      output_shape.AddDim(values_shape.dim_size(i));
    }

    Tensor *output_t;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output_t));

    const int num_output = output_t->NumElements();
    for (int i = 0; i < num_output; ++i) {
      output_t->flat<T>()(i) = default_value;
    }

    const int prefix_rank = dense_prefix_t.NumElements();
    const int nnz = values_shape.dim_size(0);
    const int element_size = values_t.NumElements() / nnz;
    for (int i = 0; i < nnz; ++i) {
      int output_idx = 0;
      for (int j = 0; j < prefix_rank; ++j) {
        output_idx = output_idx * dense_prefix_t.flat<Tshape>()(j) +
                     indices_t.flat<Tindices>()(i * prefix_rank + j);
      }

      for (int j = 0; j < element_size; ++j) {
        output_t->flat<T>()(output_idx * element_size + j) =
            values_t.flat<T>()(i * element_size + j);
      }
    }
  }
};

} // namespace functor

#define BUILD_HELPER(T, Tindices, Tshape)                                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Addons>ExtendedSparseToDense")                                     \
          .Device(tensorflow::DEVICE_CPU)                                      \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<Tindices>("Tindices")                                \
          .TypeConstraint<Tshape>("Tshape"),                                   \
      ExtendedSparseToDenseOp<CPUDevice, T, Tindices, Tshape>);

BUILD_HELPER(float, int32, int32);
BUILD_HELPER(float, int32, int64);
BUILD_HELPER(float, int64, int32);
BUILD_HELPER(float, int64, int64);

#undef BUILD_HELPER

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("Addons>ExtendedSparseToDense")
    .Input("indices: Tindices")
    .Input("values: T")
    .Input("dense_prefix: Tshape")
    .Output("dense_tensor: T")
    .Attr("default_float: float = 0.0")
    .Attr("default_int32: int = 0")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Tshape: {int32, int64} = DT_INT64")
    .Attr("T: type = DT_FLOAT")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle indices_shape = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(indices_shape, 2, &indices_shape));
      ShapeHandle values_shape = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(values_shape, 1, &values_shape));
      ShapeHandle dense_prefix;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &dense_prefix));

      ShapeHandle output_shape;
      if (c->Rank(values_shape) > 1) {
        ShapeHandle element_shape;
        TF_RETURN_IF_ERROR(c->Subshape(values_shape, 1, &element_shape));
        TF_RETURN_IF_ERROR(
            c->Merge(dense_prefix, element_shape, &output_shape));
      } else {
        output_shape = dense_prefix;
      }

      c->set_output(0, output_shape);

      return Status::OK();
    });

} // namespace feature_opt
} // namespace tensorflow