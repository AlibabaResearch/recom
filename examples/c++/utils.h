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
#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {
namespace benchmark {

int GenRandomCol();

int GenRandomInt();

double GenRandomFP();

std::string GenRandomStr();

std::string GenRandonStrWithDelim(const std::string& delim);

std::string GenRandonStrWithDelim(const std::string& delim, int ncols);

void LoadSavedModelWrap(const std::string& model_path, int intra_threads,
                        int inter_threads,
                        std::shared_ptr<SavedModelBundle>& bundle,
                        bool use_gpu = true);

void LoadOpLibrary(const std::string& lib_path);

std::string ExtractNodeByTensor(const std::string& tname);

void ConstructGraphInputs(const GraphDef& gd, int batch_size,
                          std::vector<std::pair<std::string, Tensor>>& inputs);

void ExtractGraphOutputs(const std::shared_ptr<SavedModelBundle>& bundle,
                         std::vector<std::string>& outputs);

void ExtractGraphConcats(const GraphDef& gd, std::vector<std::string>& outputs);

}  // namespace benchmark
}  // namespace tensorflow