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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stack>

#include "fc_optimizer_base.h"
#include "tensorflow_addons/utils.h"

namespace tensorflow {
namespace feature_opt {

class UselessNodesPruner : public FCOptimizerBase {
public:
  UselessNodesPruner(GraphInfo &ginfo) : FCOptimizerBase(ginfo) {}

  void Optimize();

private:
  bool MatchIdentity(NodeDef *node, int &useful_idx);

  bool MatchUselessTranspose(NodeDef *node, int &useful_idx);

  bool MatchUselessArithm(NodeDef *node, int &useful_idx);

  bool MatchUselessStridedSlice(NodeDef *node, int &useful_idx);
};

} // namespace feature_opt
} // namespace tensorflow