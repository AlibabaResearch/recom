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

#define EIGEN_USE_GPU

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cub/cub.cuh>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_requires.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/platform/errors.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "concat_outputs_ops.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

template <typename T>
ConcatOutputsOp<T>::ConcatOutputsOp(OpKernelConstruction *c) : OpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("N", &N));
  OP_REQUIRES_OK(c, c->GetAttr("BLOCK_THREADS", &BLOCK_THREADS));
  OP_REQUIRES_OK(c, c->GetAttr("embedd_dims", &embedd_dims));
  OP_REQUIRES_OK(c, c->GetAttr("device_input_indices", &device_input_indices));
  OP_REQUIRES_OK(c,
                 c->GetAttr("device_concat_indices", &device_concat_indices));
  OP_REQUIRES_OK(c, c->GetAttr("host_concat_indices", &host_concat_indices));
  OP_REQUIRES_OK(c, c->GetAttr("prefix_begin", &prefix_begin));
  OP_REQUIRES_OK(c, c->GetAttr("prefix_end", &prefix_end));
  OP_REQUIRES(c, N == host_concat_indices.size(),
              errors::InvalidArgument("N != host_concat_indices.size()"));
  OP_REQUIRES(
      c, device_input_indices.size() == device_concat_indices.size(),
      errors::InvalidArgument(
          "device_input_indices.size() != device_concat_indices.size()"));
  OP_REQUIRES(c,
              embedd_dims.size() ==
                  device_concat_indices.size() + host_concat_indices.size(),
              errors::InvalidArgument(
                  "embedd_dims.size() != device_concat_indices.size() + "
                  "host_concat_indices.size()"));

  std::string type_string;
  if (std::is_same<T, float>::value) {
    type_string = "float";
  } else if (std::is_same<T, int32>::value) {
    type_string = "int";
  } else {
    LOG(ERROR) << "Unsupported type! suppose float";
    type_string = "float";
  }

  std::vector<int> output_scans(embedd_dims.size() + 1);
  output_scans[0] = 0;
  for (int i = 0; i < embedd_dims.size(); ++i) {
    output_scans[i + 1] = output_scans[i] + embedd_dims[i];
  }
  embedd_dim_sum = output_scans.back();

  std::string code;
  code += "#include <cstdio>\n";
  code += "\n";

  code += "template <int EmbeddDim, int Offset>\n"
          "__device__ __forceinline__ void ScatterBlock(const " +
          type_string + " *__restrict__ d_inputs, " + type_string +
          " *__restrict__ d_outputs, int prefix_size) {\n"
          "  for (int i = threadIdx.x; i < prefix_size * EmbeddDim; i += " +
          std::to_string(BLOCK_THREADS) +
          ") {\n"
          "    const int prefix_idx = i / EmbeddDim;\n"
          "    const int embedd_idx = i - prefix_idx * EmbeddDim;\n"
          "    const int output_idx = prefix_idx * " +
          std::to_string(embedd_dim_sum) +
          " + Offset + embedd_idx;\n"
          "    d_outputs[output_idx] = d_inputs[i];\n"
          "  }\n"
          "}\n";
  code += "\n";

  code += "__global__ void ConcatOutputsKnl(" + type_string +
          " **device_input_ptrs, " + type_string + " **d_host_input_ptrs, " +
          type_string + " *d_outputs, int prefix_size) {\n";
  code += "  switch (blockIdx.x) {\n";
  for (int i = 0; i < device_concat_indices.size(); ++i) {
    int concat_idx = device_concat_indices[i];
    int input_idx = device_input_indices[i];
    code += "  case " + std::to_string(concat_idx) + ":\n";

    int embedd_dim = embedd_dims[concat_idx];
    int offset = output_scans[concat_idx];
    code += "    ScatterBlock<" + std::to_string(embedd_dim) + ", " +
            std::to_string(offset) + ">(device_input_ptrs[" +
            std::to_string(input_idx) + "], d_outputs, prefix_size);\n";
    code += "    break;\n";
  }

  for (int i = 0; i < host_concat_indices.size(); ++i) {
    int concat_idx = host_concat_indices[i];
    code += "  case " + std::to_string(concat_idx) + ":\n";

    int embedd_dim = embedd_dims[concat_idx];
    int offset = output_scans[concat_idx];
    code += "    ScatterBlock<" + std::to_string(embedd_dim) + ", " +
            std::to_string(offset) + ">(d_host_input_ptrs[" +
            std::to_string(i) + "], d_outputs, prefix_size);\n";
    code += "    break;\n";
  }
  code += "  }\n";
  code += "}\n\n";

  code += "extern \"C\" void ConcatOutputs(" + type_string +
          " **device_input_ptrs, " + type_string + " **d_host_input_ptrs, " +
          type_string + " *d_outputs, int prefix_size, cudaStream_t strm) {\n";
  code += "  ConcatOutputsKnl<<<" + std::to_string(embedd_dims.size()) + ", " +
          std::to_string(BLOCK_THREADS) +
          ", 0, strm>>>(device_input_ptrs, d_host_input_ptrs, d_outputs, "
          "prefix_size);\n";
  code += "}\n";

  std::string output_dir;
  OP_REQUIRES_OK(c, c->GetAttr("output_dir", &output_dir));

  bool debug_mode = GetEnv("RECOM_DEBUG", "off") == "on";
  const std::string &code_md5 = GetStringMD5(code);
  dlpath = output_dir + "/" + code_md5 + ".so";

  if (!ExistFile(dlpath)) {
    const std::string &code_path =
        output_dir + "/" + code_md5 + (debug_mode ? "_debug" : "") + ".cu";
    std::ofstream code_file(code_path);
    code_file << code << std::endl;

    std::string compile_cmd = "nvcc -Xcompiler -fPIC --shared -o " + dlpath +
                              " " + code_path +
                              (debug_mode ? " -O0 -G -g" : " -O3");

    if (GetEnv("RECOM_FORMAT_GEN_CODES", "off") == "on") {
      if (system(("clang-format -i " + code_path).c_str()) != 0) {
        LOG(WARNING) << "Fail to format the codes. Please ensure clang-format "
                        "is installed.";
      }
    }

    OP_REQUIRES(c, system(compile_cmd.c_str()) == 0,
                errors::Aborted("Fail to compile ", code_path, " to ", dlpath));
    LOG(INFO) << "Compile " << code_path << " to " << dlpath << " successfully";
  } else {
    LOG(INFO) << dlpath << " already exists in the cache!";
  }

  void *handle = dlopen(dlpath.c_str(), RTLD_NOW);
  OP_REQUIRES(c, handle, errors::Aborted("Fail to dlopen ", dlpath));

  ConcatOutputs =
      reinterpret_cast<decltype(ConcatOutputs)>(dlsym(handle, "ConcatOutputs"));
}

template <typename T> void ConcatOutputsOp<T>::Compute(OpKernelContext *c) {
  int num_host_input_elements = 0;
  for (int i = 0; i < N; ++i) {
    num_host_input_elements += c->input(i + 2).NumElements();
  }

  Tensor host_inputs_t;
  OP_REQUIRES_OK(
      c, c->allocate_temp(
             DT_INT8, {static_cast<long>(num_host_input_elements * sizeof(T))},
             &host_inputs_t));
  T *d_host_inputs = reinterpret_cast<T *>(host_inputs_t.data());

  std::vector<T *> h_host_input_ptrs(N);
  std::vector<T> h_host_inputs(num_host_input_elements);
  for (int i = 0, idx = 0; i < N; ++i) {
    int n = c->input(i + 2).NumElements();
    memcpy(&h_host_inputs[idx], c->input(i + 2).data(), n * sizeof(T));
    h_host_input_ptrs[i] = d_host_inputs + idx;
    idx += n;
  }

  cudaStream_t strm = c->eigen_gpu_device().stream();

  CubDebugExit(cudaMemcpyAsync(d_host_inputs, &h_host_inputs[0],
                               h_host_inputs.size() * sizeof(T),
                               cudaMemcpyHostToDevice, strm));

  Tensor host_input_ptrs_t;
  OP_REQUIRES_OK(c,
                 c->allocate_temp(DT_INT8, {static_cast<long>(N * sizeof(T *))},
                                  &host_input_ptrs_t));
  T **d_host_input_ptrs = reinterpret_cast<T **>(host_input_ptrs_t.data());

  CubDebugExit(cudaMemcpyAsync(d_host_input_ptrs, &h_host_input_ptrs[0],
                               h_host_input_ptrs.size() * sizeof(T *),
                               cudaMemcpyHostToDevice, strm));

  const int *device_input_shapes = c->input(1).flat<int>().data();
  int prefix_size = 1;
  TensorShape output_shape;
  for (int i = prefix_begin; i < prefix_end; ++i) {
    int dim_size = device_input_shapes[i];
    output_shape.AddDim(dim_size);
    prefix_size *= dim_size;
  }
  output_shape.AddDim(embedd_dim_sum);

  Tensor *output_t;
  OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output_t));

  ConcatOutputs(reinterpret_cast<T **>(c->input(0).data()), d_host_input_ptrs,
                output_t->flat<T>().data(), prefix_size, strm);

  CubDebugExit(cudaStreamSynchronize(strm));
}

#define REGISTER_CONCAT(T)                                                     \
  REGISTER_KERNEL_BUILDER(Name("Addons>ConcatOutputs")                         \
                              .Device(tensorflow::DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")                          \
                              .HostMemory("host_inputs")                       \
                              .HostMemory("device_input_shapes"),              \
                          ConcatOutputsOp<T>);                                 \
  REGISTER_KERNEL_BUILDER(Name("Addons>ConcatOutputsNoHost")                   \
                              .Device(tensorflow::DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")                          \
                              .HostMemory("device_input_shapes"),              \
                          ConcatOutputsOp<T>);

REGISTER_CONCAT(float);
REGISTER_CONCAT(int);

#undef REGISTER_CONCAT

REGISTER_OP("Addons>ConcatOutputs")
    .Input("device_input_ptrs: int64")
    .Input("device_input_shapes: int32")
    .Input("host_inputs: N * T")
    .Input("tensor_buffers: buffer_types")
    .Output("output: T")
    .Attr("T: type")
    .Attr("N: int")
    .Attr("embedd_dims: list(int)")
    .Attr("device_input_indices: list(int)")
    .Attr("device_concat_indices: list(int)")
    .Attr("host_concat_indices: list(int)")
    .Attr("prefix_begin: int")
    .Attr("prefix_end: int")
    .Attr("buffer_types: list(type)")
    .Attr("BLOCK_THREADS: int")
    .Attr("output_dir: string");

REGISTER_OP("Addons>ConcatOutputsNoHost")
    .Input("device_input_ptrs: int64")
    .Input("device_input_shapes: int32")
    .Input("tensor_buffers: buffer_types")
    .Output("output: T")
    .Attr("T: type")
    .Attr("N: int")
    .Attr("embedd_dims: list(int)")
    .Attr("device_input_indices: list(int)")
    .Attr("device_concat_indices: list(int)")
    .Attr("host_concat_indices: list(int)")
    .Attr("prefix_begin: int")
    .Attr("prefix_end: int")
    .Attr("buffer_types: list(type)")
    .Attr("BLOCK_THREADS: int")
    .Attr("output_dir: string");

} // namespace feature_opt
} // namespace tensorflow