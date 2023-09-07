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
#include <vector>

#include "extended_sparse_segment_reduce_ops.h"

namespace tensorflow {
namespace feature_opt {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T, typename Tindices, typename Tspvalues,
          typename Tshape>
class ExtendedSparseSegmentMeanOp : public OpKernel {
public:
  explicit ExtendedSparseSegmentMeanOp(OpKernelConstruction *c) : OpKernel(c) {}

  void Compute(OpKernelContext *c) override {
    functor::ExtendedSparseSegmentMeanFunctor<Device, T, Tindices, Tspvalues,
                                              Tshape>()(
        c, c->input(0), c->input(1), c->input(2), c->input(3), c->input(4));
  }
};

namespace functor {

template <typename T, typename Tindices, typename Tspvalues, typename Tshape>
class ExtendedSparseSegmentMeanFunctor<CPUDevice, T, Tindices, Tspvalues,
                                       Tshape> {
public:
  void operator()(OpKernelContext *c, const Tensor &weight_t,
                  const Tensor &sp_indices_t, const Tensor &sp_values_t,
                  const Tensor &dense_shape_t, const Tensor &dense_prefix_t) {
    const int num_input = sp_values_t.NumElements();
    const int input_rank = dense_shape_t.NumElements();
    const int output_rank = dense_prefix_t.NumElements();
    const int embedd_dim = weight_t.dim_size(1);

    std::vector<int> segment_ids(num_input);
    for (int i = 0; i < num_input; ++i) {
      int segment_id = 0;
      for (int j = 0; j < input_rank - 1; ++j) {
        segment_id = segment_id * dense_shape_t.flat<Tshape>()(j) +
                     sp_indices_t.flat<Tindices>()(i * input_rank + j);
      }
      segment_ids[i] = segment_id;
    }

    int num_output = 1;
    for (int i = 1; i < num_input; ++i) {
      if (segment_ids[i] > segment_ids[i - 1])
        ++num_output;
    }

    Tensor *exsp_indices_t;
    OP_REQUIRES_OK(
        c, c->allocate_output(0, {num_output, output_rank}, &exsp_indices_t));

    Tensor *exsp_values_t;
    OP_REQUIRES_OK(
        c, c->allocate_output(1, {num_output, embedd_dim}, &exsp_values_t));
    memset(exsp_values_t->data(), 0, exsp_values_t->NumElements() * sizeof(T));

    std::vector<int> dense_prefix_offsets(output_rank, 1);
    for (int i = output_rank - 2; i >= 0; --i) {
      dense_prefix_offsets[i] =
          dense_prefix_offsets[i + 1] * dense_prefix_t.flat<Tshape>()(i + 1);
    }

    for (int i = 0, cnt = 0, out_idx = 0; i < num_input; ++i) {
      const int lookup_idx = sp_values_t.flat<Tspvalues>()(i);
      for (int j = 0; j < embedd_dim; ++j) {
        exsp_values_t->flat<T>()(out_idx * embedd_dim + j) +=
            weight_t.flat<T>()(lookup_idx * embedd_dim + j);
      }
      ++cnt;

      if (i + 1 == num_input || segment_ids[i + 1] > segment_ids[i]) {
        int segment_id = segment_ids[i];
        for (int j = 0; j < output_rank; ++j) {
          int idx = segment_id / dense_prefix_offsets[j];
          exsp_indices_t->flat<Tindices>()(out_idx * output_rank + j) = idx;
          segment_id -= idx * dense_prefix_offsets[j];
        }

        for (int j = 0; j < embedd_dim; ++j) {
          exsp_values_t->flat<T>()(out_idx * embedd_dim + j) /= cnt;
        }
        ++out_idx;
        cnt = 0;
      }
    }
  }
};

} // namespace functor

#define BUILD_HELPER(T, Tindices, Tspvalues, Tshape)                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Addons>ExtendedSparseSegmentMean")                                 \
          .Device(tensorflow::DEVICE_CPU)                                      \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<Tindices>("Tindices")                                \
          .TypeConstraint<Tspvalues>("Tspvalues")                              \
          .TypeConstraint<Tshape>("Tshape"),                                   \
      ExtendedSparseSegmentMeanOp<CPUDevice, T, Tindices, Tspvalues, Tshape>);

BUILD_HELPER(float, int32, int32, int32);
BUILD_HELPER(float, int32, int32, int64);
BUILD_HELPER(float, int32, int64, int32);
BUILD_HELPER(float, int32, int64, int64);
BUILD_HELPER(float, int64, int32, int32);
BUILD_HELPER(float, int64, int32, int64);
BUILD_HELPER(float, int64, int64, int32);
BUILD_HELPER(float, int64, int64, int64);

#undef BUILD_HELPER

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("Addons>ExtendedSparseSegmentMean")
    .Input("weight: T")
    .Input("sp_indices: Tindices")
    .Input("sp_values: Tspvalues")
    .Input("dense_shape: Tshape")
    .Input("dense_prefix: Tshape")
    .Output("exsp_indices: Tindices")
    .Output("exsp_values: T")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Tspvalues: {int32, int64} = DT_INT64")
    .Attr("Tshape: {int32, int64} = DT_INT64")
    .Attr("T: type = DT_FLOAT")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle weight = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(weight, 2, &weight));
      ShapeHandle dense_prefix = c->input(4);
      TF_RETURN_IF_ERROR(c->WithRank(dense_prefix, 1, &dense_prefix));

      DimensionHandle output_rank = c->NumElements(dense_prefix);
      DimensionHandle embedd_dim = c->Dim(weight, 1);
      ShapeHandle output_indices =
          c->Matrix(InferenceContext::kUnknownDim, output_rank);
      ShapeHandle output_values =
          c->Matrix(InferenceContext::kUnknownDim, embedd_dim);
      c->set_output(0, output_indices);
      c->set_output(1, output_values);
      return Status::OK();
    });

} // namespace feature_opt
} // namespace tensorflow