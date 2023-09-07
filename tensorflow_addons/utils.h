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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

std::string GetEnv(const std::string &key, const std::string &default_val);

bool ExistFile(const std::string &fname);

std::string GetStringMD5(const std::string &s);

#define HashSetT std::unordered_set
#define HashMapT std::unordered_map

using NodeMap = HashMapT<std::string, NodeDef *>;
using OutMap = HashMapT<std::string, HashSetT<std::string>>;

std::string GetEmbeddingNamePrefix(const std::string &name);

std::string GetNameOuterPrefix(const std::string &name);

std::string GetNameInnerPrefix(const std::string &name);

std::string GetNodeNameByTensor(const std::string &tensor_name);

int GetOutputIdxByTensor(const std::string &tensor_name);

std::string FormTensorName(const NodeDef *node, int index);

std::vector<std::vector<int>> FetchGrapplerOutputShapes(NodeDef *node);

const int recom_log_verbosity = std::stoi(GetEnv("RECOM_LOG_VERBOSITY", "0"));
#define RECOM_LOG_IF(LEVEL, CONDITION)                                         \
  if (CONDITION)                                                               \
  LOG(LEVEL)
#define RECOM_VLOG RECOM_LOG_IF(INFO, 2 <= recom_log_verbosity)
#define RECOM_VLOG_WARNING RECOM_LOG_IF(INFO, 1 <= recom_log_verbosity)

#define RETURN_FALSE                                                           \
  {                                                                            \
    RECOM_VLOG << "Failure";                                                   \
    return false;                                                              \
  }

#define LOG_AND_RETURN_FALSE(msg)                                              \
  {                                                                            \
    RECOM_VLOG << msg;                                                         \
    return false;                                                              \
  }

#define RETURN_IF_FALSE(statement)                                             \
  {                                                                            \
    if (!(statement)) {                                                        \
      RECOM_VLOG << #statement << " false";                                    \
      return false;                                                            \
    }                                                                          \
  }

#define LOG_AND_RETURN_IF_FALSE(statement, msg)                                \
  {                                                                            \
    if (!(statement)) {                                                        \
      RECOM_VLOG << msg;                                                       \
      return false;                                                            \
    }                                                                          \
  }

} // namespace feature_opt
} // namespace tensorflow