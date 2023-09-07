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

#include "fc_optimize_pass.h"
#include "tensorflow_addons/utils.h"
#include "tensorflow_addons/graph_optimizers/cuda_emitter.h"
#include "tensorflow_addons/graph_optimizers/lookup_optimizer.h"
#include "tensorflow_addons/graph_optimizers/post_lookup_optimizer.h"
#include "tensorflow_addons/graph_optimizers/pre_lookup_optimizer.h"
#include "tensorflow_addons/graph_optimizers/shape_construct_optimizer.h"
#include "tensorflow_addons/graph_optimizers/useless_nodes_pruner.h"

#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace feature_opt {

Status FCOptimizePass::Run(const GraphOptimizationPassOptions &options) {
  grappler::GrapplerItem item;
  item.id = "tf_graph";
  (*options.graph)->ToGraphDef(&item.graph);

  tensorflow::DumpGraphDefToFile("before_opt", item.graph);

  GraphInfo graph_info(item);

  LOG(INFO) << "Start Useless Pruner";
  UselessNodesPruner useless_pruner(graph_info);
  useless_pruner.Optimize();
  graph_info.UpdateTopoInfo();

  ShapeConstructOptimizer shape_construct_opt(graph_info);
  if (GetEnv("RECOM_SHAPE_OPT", "on") != "off") {
    LOG(INFO) << "Start Shape Construct Optimization";
    shape_construct_opt.OptimizeShapeInputs();
    graph_info.UpdateAll();
  }

  if (GetEnv("RECOM_EMBEDDING_COLUMN_OPT", "on") != "off") {
    if (GetEnv("RECOM_PRE_LOOKUP_OPT", "on") != "off") {
      LOG(INFO) << "Start Pre-Lookup Optimization";
      PreLookupOptimizer pre_lookup_opt(graph_info);
      pre_lookup_opt.Optimize();
      graph_info.UpdateAll();
    }

    if (GetEnv("RECOM_LOOKUP_OPT", "on") != "off") {
      LOG(INFO) << "Start Lookup Optimization";
      LookupOptimizer lookup_opt(graph_info);
      lookup_opt.Optimize();
      graph_info.UpdateAll();
    }
  }

  if (GetEnv("RECOM_SHAPE_OPT", "on") != "off") {
    LOG(INFO) << "Start Shape Construct Optimization (Pruning)";
    shape_construct_opt.PruneDeadShapeConstruct();
    graph_info.UpdateAll();
  }

  if (GetEnv("RECOM_CODEGEN", "on") != "off") {
    LOG(INFO) << "Start CUDA Emitter";
    CudaEmitter cuda_emitter(graph_info, 1 << 28, 64);
    cuda_emitter.Optimize();
  }

  tensorflow::DumpGraphDefToFile("after_opt", item.graph);

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  // opts.expect_device_spec = true;
  auto optimized_graph = std::make_unique<Graph>(OpRegistry::Global());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(opts, item.graph, optimized_graph.get()));
  *options.graph = std::move(optimized_graph);

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      FCOptimizePass);

} // namespace feature_opt
} // namespace tensorflow