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
#include <boost/icl/interval.hpp>
#include <boost/icl/interval_set.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stack>

#include "fc_optimizer_base.h"

namespace tensorflow {
namespace feature_opt {

class PreLookupOptimizer : public FCOptimizerBase {
public:
  PreLookupOptimizer(GraphInfo &ginfo) : FCOptimizerBase(ginfo) {}

  using IntervalSet = boost::icl::interval_set<int>;
  using Interval = IntervalSet::interval_type;
  using IntervalBounds = boost::icl::interval_bounds;

  static const IntervalSet UniversalSet;

private:
  // TODO: consider the type casting. may need to record it on meta

  struct OpMeta {
    enum Type { Gather, Select, Keep, Map };
    Type type;
    IntervalSet s;
    int a;

    OpMeta() = default;
    OpMeta(Type type) : type(type), a(0) { s = UniversalSet; }
    OpMeta(Type type, const IntervalSet &s, int a = 0)
        : type(type), s(s), a(a) {}
  };

  std::stack<NodeDef *> indice_reserve_ops;
  std::stack<NodeDef *> value_reserve_ops;
  std::vector<OpMeta> meta_vec;

public:
  void Optimize();

private:
  bool Optimize(NodeDef *sfer);

  bool MatchExpr(const NodeDef *op_node, NodeDef *&in_value_node,
                 int &in_value_out_idx, IntervalSet &s);

  bool MatchGatherValue(NodeDef *&value_node, int &value_out_idx,
                        NodeDef *&indice_node, int &indice_out_idx);

  bool MatchGatherValue(NodeDef *&value_node, int &value_out_idx);

  bool MatchSelectValue(NodeDef *&value_node, int &value_out_idx);

  bool MatchMapValue(NodeDef *&value_node, int &value_out_idx);

  bool MatchKeepValue(NodeDef *&value_node, int &value_out_idx);

  bool MatchSourceValue(NodeDef *&value_node, int &value_out_idx,
                        NodeDef *&indice_node, int &indice_out_idx,
                        DataType &indice_type);

  void SkipReserveIndiceOps(NodeDef *&indice_node, int &indice_out_idx);

  bool MatchBeforeLookup(NodeDef *&value_node, int &value_out_idx,
                         NodeDef *&indice_node, int &indice_out_idx,
                         DataType &indice_type);

  bool Simplify();

  std::vector<std::pair<int, int>> GetClosedBoundaries(const IntervalSet &s);

  bool ReconstructGraph(NodeDef *sfer, NodeDef *value_node, int value_out_idx,
                        NodeDef *indice_node, int indice_out_idx,
                        DataType init_indice_type);

  bool ReconstructKeepValue(NodeDef *&value_node, int &value_out_idx,
                            const OpMeta &meta);

  bool ReconstructMapValue(NodeDef *&value_node, int &value_out_idx,
                           const OpMeta &meta);

  bool ReconstructGatherValue(NodeDef *&value_node, int &value_out_idx,
                              NodeDef *&indice_node, int &indice_out_idx,
                              const OpMeta &meta);

  bool ReconstructSelectValue(NodeDef *&value_node, int &value_out_idx,
                              const OpMeta &meta);

  bool ReconstructIndice(NodeDef *&indice_node, int &indice_out_idx);

  bool ReconnectToSFER(NodeDef *sfer, NodeDef *&value_node, int &value_out_idx,
                       NodeDef *&indice_node, int &indice_out_idx);
};

} // namespace feature_opt
} // namespace tensorflow