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

class LookupOptimizer : public FCOptimizerBase {
public:
  LookupOptimizer(GraphInfo &ginfo) : FCOptimizerBase(ginfo) {}

  void Optimize();

private:
  struct DominantNodes {
    NodeDef *seed;
    NodeDef *slice;
    NodeDef *sfer;
    NodeDef *weight;
    NodeDef *select;
  };

  bool Match(NodeDef *sparse_segment_node, DominantNodes &dominant);

  bool MatchDenseInput(DominantNodes dominant);

  bool MatchGatherScatter(DominantNodes dominant);

  bool RewriteSeedWithNumSegments(DominantNodes dominant, int fc_id);

  // deprecated
  bool RewriteExtendedSparse(DominantNodes dominant, int fc_id);

  bool RewriteDenseInput(DominantNodes dominant, int fc_id);

  bool RewriteGatherScatter(DominantNodes dominant, int fc_id);
};

} // namespace feature_opt
} // namespace tensorflow