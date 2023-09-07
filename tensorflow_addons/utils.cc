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

#include "utils.h"
#include "openssl/md5.h"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace tensorflow {
namespace feature_opt {

std::string GetEnv(const std::string &key, const std::string &default_val) {
  char *env = getenv(key.c_str());
  return env ? env : default_val;
}

bool ExistFile(const std::string &fname) {
  std::ifstream ist(fname);
  return ist.good();
}

std::string GetStringMD5(const std::string &s) {
  uint8_t md5_output[16];
  MD5(reinterpret_cast<const uint8_t *>(s.c_str()), s.size(),
      reinterpret_cast<uint8_t *>(md5_output));

  std::stringstream ss;
  for (int i = 0; i < 16; ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<int>(md5_output[i]);
  }
  return ss.str();
}

std::string GetEmbeddingNamePrefix(const std::string &name) {
  size_t pos = 0;
  std::string prefix;
  while (pos != std::string::npos) {
    pos = name.find("/", pos + 1);
    prefix = name.substr(0, pos);
    if (prefix.find("embedding") != std::string::npos)
      break;
  }

  if (pos != std::string::npos) {
    return prefix;
  } else {
    return GetNameInnerPrefix(name);
  }
}

std::string GetNameOuterPrefix(const std::string &name) {
  return name.substr(0, name.find("/"));
}

std::string GetNameInnerPrefix(const std::string &name) {
  return name.substr(0, name.find_last_of("/"));
}

std::string GetNodeNameByTensor(const std::string &tensor_name) {
  std::vector<std::string> parts =
      absl::StrSplit(tensor_name, absl::ByAnyChar("^:"), absl::SkipEmpty());
  return parts.empty() ? "" : parts[0];
}

int GetOutputIdxByTensor(const std::string &tensor_name) {
  std::vector<std::string> parts =
      absl::StrSplit(tensor_name, absl::ByAnyChar("^:"), absl::SkipEmpty());
  return parts.size() == 2 ? std::stoi(parts[1]) : 0;
}

std::string FormTensorName(const NodeDef *node, int index) {
  if (index == 0)
    return node->name();
  return node->name() + ":" + std::to_string(index);
}

std::vector<std::vector<int>> FetchGrapplerOutputShapes(NodeDef *node) {
  auto output_shapes = node->attr().at("_output_shapes").list();
  std::vector<std::vector<int>> results(output_shapes.shape_size());
  for (int i = 0; i < output_shapes.shape_size(); ++i) {
    auto shape = output_shapes.shape(i);
    results[i] = std::vector<int>(shape.dim_size());
    for (int j = 0; j < shape.dim_size(); ++j) {
      results[i][j] = shape.dim(j).size();
    }
  }
  return results;
}

} // namespace feature_opt
} // namespace tensorflow