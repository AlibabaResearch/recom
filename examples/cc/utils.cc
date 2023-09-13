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

#include <cstdlib>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_replace.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "utils.h"

namespace tensorflow {
namespace benchmark {

int GenRandomCol() { return rand() % 10 + 1; }

int GenRandomInt() { return rand() % 100; }

double GenRandomFP() {
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * 100;
}

std::string GenRandomStr() {
  std::string raw = std::to_string(GenRandomFP());
  return raw.substr(0, raw.find(".") + 5);
}

std::string GenRandonStrWithDelim(const std::string& delim) {
  std::vector<std::string> elements(GenRandomCol());
  std::generate(elements.begin(), elements.end(), GenRandomStr);
  return absl::StrJoin(elements, delim);
}

std::string GenRandonStrWithDelim(const std::string& delim, int ncols) {
  std::vector<std::string> elements(ncols);
  std::generate(elements.begin(), elements.end(), GenRandomStr);
  return absl::StrJoin(elements, delim);
}

std::string ExtractNodeByTensor(const std::string& tname) {
  std::string replaced = absl::StrReplaceAll(tname, {{"^", ""}});
  std::vector<std::string> splitted = absl::StrSplit(replaced, ":");
  return splitted.size() > 0 ? splitted[0] : "";
}

void ConstructGraphInputs(const GraphDef& gd, int batch_size,
                          std::vector<std::pair<std::string, Tensor>>& inputs) {
  std::unordered_map<std::string, const NodeDef*> node_mapping;
  for (const NodeDef& node : gd.node()) {
    node_mapping[node.name()] = &node;
  }

  std::vector<int> batch_ncols(batch_size);
  std::generate(batch_ncols.begin(), batch_ncols.end(), GenRandomCol);

  std::unordered_map<std::string, Tensor> input_dict;
  for (const NodeDef& node : gd.node()) {
    if (node.op() == "StringSplitV2" || node.op() == "StringSplit") {
      const NodeDef* delim_node =
          node_mapping.at(ExtractNodeByTensor(node.input(1)));
      std::string delim = delim_node->attr().at("value").tensor().string_val(0);
      // LOG(INFO) << "StringSplit Node: " << delim_node->name()
      //           << ", Delimeter: " << delim;

      std::queue<const NodeDef*> node_queue;
      node_queue.push(node_mapping.at(ExtractNodeByTensor(node.input(0))));
      while (!node_queue.empty()) {
        const NodeDef* n = node_queue.front();
        node_queue.pop();

        if (n->op() == "Placeholder") {
          Tensor t(DT_STRING, {batch_size});
          for (int i = 0; i < batch_size; ++i) {
            t.flat<tstring>()(i) = GenRandonStrWithDelim(delim, batch_ncols[i]);
          }
          input_dict[n->name() + ":0"] = t;
        }

        for (const std::string& p : n->input()) {
          node_queue.push(node_mapping.at(ExtractNodeByTensor(p)));
        }
      }
    }
  }

  for (const NodeDef& node : gd.node()) {
    if (node.op() == "Placeholder" && !input_dict.count(node.name() + ":0")) {
      const auto& shape_ptoto = node.attr().at("shape").shape();
      TensorShape shape;
      for (int i = 0; i < shape_ptoto.dim_size(); ++i) {
        if (shape_ptoto.dim(i).size() < 0) {
          shape.AddDim(batch_size);
        } else {
          shape.AddDim(shape_ptoto.dim(i).size());
        }
      }
      DataType dtype = node.attr().at("dtype").type();
      Tensor t(dtype, shape);
      switch (dtype) {
        case DT_FLOAT: {
          for (int i = 0; i < t.NumElements(); ++i) {
            t.flat<float>()(i) = GenRandomFP();
          }
        } break;
        case DT_DOUBLE: {
          for (int i = 0; i < t.NumElements(); ++i) {
            t.flat<double>()(i) = GenRandomFP();
          }
        } break;
        case DT_INT32: {
          for (int i = 0; i < t.NumElements(); ++i) {
            t.flat<int>()(i) = GenRandomInt();
          }
        } break;
        case DT_INT64: {
          for (int i = 0; i < t.NumElements(); ++i) {
            t.flat<int64>()(i) = GenRandomInt();
          }
        } break;
        case DT_STRING: {
          for (int i = 0; i < t.NumElements(); ++i) {
            t.flat<tstring>()(i) = GenRandomStr();
          }
        } break;
        default:
          LOG(ERROR) << "Not support the input type " << DataTypeString(dtype);
          exit(-1);
      }
      input_dict[node.name() + ":0"] = t;
    }
  }

  inputs = std::vector<std::pair<std::string, Tensor>>(input_dict.begin(),
                                                       input_dict.end());
}

void LoadSavedModelWrap(const std::string& model_path, int intra_threads,
                        int inter_threads,
                        std::shared_ptr<SavedModelBundle>& bundle,
                        bool use_gpu) {
  std::unordered_set<std::string> tags = {"serve"};
  bundle.reset(new SavedModelBundle());
  auto options = SessionOptions();
  options.config.set_intra_op_parallelism_threads(intra_threads);
  options.config.set_inter_op_parallelism_threads(inter_threads);
  if (!use_gpu) (*options.config.mutable_device_count())["GPU"] = 0;
  auto* gpu_opt = options.config.mutable_gpu_options();
  gpu_opt->set_per_process_gpu_memory_fraction(0.7);
  gpu_opt->set_allow_growth(true);

  LOG(INFO) << "Start load model";
  auto t1 = std::chrono::high_resolution_clock::now();
  Status status =
      LoadSavedModel(options, RunOptions(), model_path, tags, bundle.get());
  if (!status.ok()) {
    LOG(ERROR) << "Loading tf model failed. dir: " << model_path
               << " status:" << status.ToString();
    exit(-1);
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  int load_time =
      std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
  LOG(INFO) << "Load model time: " << load_time << "s";
}

void ExtractGraphOutputs(const std::shared_ptr<SavedModelBundle>& bundle,
                         std::vector<std::string>& outputs) {
  outputs = std::vector<std::string>();
  for (auto signature_item : bundle->meta_graph_def.signature_def()) {
    if (signature_item.first == kDefaultServingSignatureDefKey) {
      SignatureDef signature = signature_item.second;
      if (signature.method_name() != kPredictMethodName &&
          signature.method_name() != kClassifyMethodName &&
          signature.method_name() != kRegressMethodName) {
        LOG(ERROR)
            << "Expected prediction signature method_name must be one of {"
            << kPredictMethodName << ", " << kClassifyMethodName << ", "
            << kRegressMethodName << "}. Was: " << signature.method_name();
      }

      for (auto output : signature.outputs()) {
        outputs.push_back(output.second.name());
      }
    }
  }
}

void ExtractGraphConcats(const GraphDef& gd,
                         std::vector<std::string>& outputs) {
  // The concat extraction logic is very simple here as we know the number of
  // embedding columns of each benchmark model is > 5, while other concat nodes
  // other than the converging node only have < 5 inputs. Do NOT rely on it to
  // extract converging node in production.
  LOG(INFO) << "Please ensure the number of embedding columns is > 5";
  outputs = std::vector<std::string>();
  for (const NodeDef& node : gd.node()) {
    if (node.op() == "ConcatV2" && node.input_size() > 5) {
      outputs.push_back(node.name() + ":0");
    }
  }
}

void LoadOpLibrary(const std::string& lib_path) {
  TF_Status* load_status = TF_NewStatus();
  TF_Library* lib_handle = TF_LoadLibrary(lib_path.c_str(), load_status);
  if (TF_GetCode(load_status) == TF_OK) {
    LOG(INFO) << "Load " << lib_path << " successfully";
  } else {
    LOG(INFO) << "Failed to load " << lib_path << ": "
              << TF_Message(load_status);
  }
  TF_DeleteStatus(load_status);
  TF_DeleteLibraryHandle(lib_handle);
}

}  // namespace benchmark
}  // namespace tensorflow