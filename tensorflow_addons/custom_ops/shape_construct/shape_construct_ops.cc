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

#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <openssl/md5.h>
#include <regex>
#include <sstream>
#include <string>
#include <symengine/expression.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_requires.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/errors.h>
#include <type_traits>
#include <unordered_map>

#include "shape_construct_ops.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

template <typename T>
ShapeConstructOp<T>::ShapeConstructOp(OpKernelConstruction *c) : OpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("Tinputs", &input_types));
  OP_REQUIRES_OK(c, c->GetAttr("symbols", &symbols));
  OP_REQUIRES_OK(c, c->GetAttr("indices", &indices));
  OP_REQUIRES_OK(c, c->GetAttr("exprs", &expr_strings));
  OP_REQUIRES(
      c,
      input_types.size() == symbols.size() && symbols.size() == indices.size(),
      errors::InvalidArgument("Attribute error! Input size inconsistent"));

  for (DataType dt : input_types) {
    OP_REQUIRES(c, dt == DT_INT32 || dt == DT_INT64,
                errors::InvalidArgument("Attribute error! Input type invalid"));
  }

  std::string output_dir = "";
  OP_REQUIRES_OK(c, c->GetAttr("output_dir", &output_dir));

  dlpath = "";
  if (output_dir != "") {
    std::string code;
    code += "#include <cmath>\n"
            "#include <vector>\n"
            "using namespace std;\n";

    code += "extern \"C\" void ShapeConstruct(const vector<long> &inputs, ";
    if (std::is_same<T, int32>::value) {
      code += "int *outputs) {\n";
    } else {
      code += "long *outputs) {\n";
    }

    std::unordered_map<std::string, int> transform_id_mapping;
    for (int i = 0; i < symbols.size(); ++i) {
      code += "  const int " + symbols[i] + " = inputs[" + std::to_string(i) +
              "];\n";
      transform_id_mapping[symbols[i]] = i;
    }

    for (int i = 0; i < expr_strings.size(); ++i) {
      code +=
          "  outputs[" + std::to_string(i) + "] = " + expr_strings[i] + ";\n";
    }
    code += "}";

    for (const auto &transform_pair : transform_id_mapping) {
      std::string before = transform_pair.first + "([^\\d])";
      std::string after = "y" + std::to_string(transform_pair.second) + "$1";
      code = std::regex_replace(code, std::regex(before), after);
    }

    bool debug_mode = GetEnv("RECOM_DEBUG", "off") == "on";
    const std::string &code_md5 = GetStringMD5(code);
    dlpath = output_dir + "/" + code_md5 + (debug_mode ? "_debug" : "") + ".so";

    if (!ExistFile(dlpath)) {
      const std::string &code_path = output_dir + "/" + code_md5 + ".cc";
      std::ofstream code_file(code_path);
      code_file << code << std::endl;

      std::string compile_cmd = "g++ " + code_path + " -fPIC -shared -o " +
                                dlpath + (debug_mode ? " -O0 -G -g" : " -O3");

      if (system(compile_cmd.c_str()) == 0) {
        LOG(INFO) << "Compile " << code_path << " to " << dlpath
                  << " successfully";
      } else {
        LOG(WARNING) << "Fail to compile " << code_path << " to " << dlpath;
        dlpath = "";
      }
    }

    if (dlpath != "") {
      void *handle = dlopen(dlpath.c_str(), RTLD_NOW);
      OP_REQUIRES(c, handle, errors::Aborted("Fail to dlopen ", dlpath));

      ShapeConstruct = (void (*)(const std::vector<int64> &, void *))dlsym(
          handle, "ShapeConstruct");
    }
  }
}

template <typename T> void ShapeConstructOp<T>::Compute(OpKernelContext *c) {
  const int num_inputs = input_types.size();
  const int output_size = expr_strings.size();

  Tensor *output_tensor;
  OP_REQUIRES_OK(
      c, c->allocate_output(0, TensorShape({output_size}), &output_tensor));

  std::vector<int64> inputs(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const Tensor &input = c->input(i);
    if (input_types[i] == DT_INT32) {
      inputs[i] = input.flat<int>()(indices[i]);
    } else {
      inputs[i] = input.flat<int64>()(indices[i]);
    }
  }

  if (dlpath == "") {
    std::vector<SymEngine::Expression> exprs(output_size);
    for (int i = 0; i < output_size; ++i)
      exprs[i] = SymEngine::Expression(expr_strings[i]);

    for (int i = 0; i < num_inputs; ++i) {
      for (int j = 0; j < output_size; ++j) {
        exprs[j] = exprs[j].subs({{SymEngine::Expression(symbols[i]),
                                   SymEngine::Expression(inputs[i])}});
      }
    }

    for (int i = 0; i < output_size; ++i) {
      output_tensor->flat<T>()(i) = static_cast<T>(exprs[i]);
    }
  } else {
    ShapeConstruct(inputs, output_tensor->data());
  }
}

REGISTER_KERNEL_BUILDER(Name("Addons>ShapeConstruct")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32>("T"),
                        ShapeConstructOp<int32>);

REGISTER_KERNEL_BUILDER(Name("Addons>ShapeConstruct")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int64>("T"),
                        ShapeConstructOp<int64>);

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("Addons>ShapeConstruct")
    .Input("inputs: Tinputs")
    .Output("output: T")
    .Attr("symbols: list(string)")
    .Attr("indices: list(int)")
    .Attr("exprs: list(string)")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("Tinputs: list(type)")
    .Attr("output_dir: string")
    .SetShapeFn([](InferenceContext *c) {
      std::vector<string> exprs;
      TF_RETURN_IF_ERROR(c->GetAttr("exprs", &exprs));
      c->set_output(0, c->Vector(exprs.size()));
      return Status::OK();
    });

} // namespace feature_opt
} // namespace tensorflow