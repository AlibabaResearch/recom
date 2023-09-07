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

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/util/dump_graph.h"

#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tensorflow {
namespace feature_opt {

class TfCpuGpuPass : public GraphOptimizationPass {
public:
  TfCpuGpuPass() {}

  Status Run(const GraphOptimizationPassOptions &options) override;

private:
  TF_DISALLOW_COPY_AND_ASSIGN(TfCpuGpuPass);
};

Status TfCpuGpuPass::Run(const GraphOptimizationPassOptions &options) {
  grappler::GrapplerItem item;
  item.id = "tf_graph";
  (*options.graph)->ToGraphDef(&item.graph);

  std::unordered_map<std::string, NodeDef *> node_mapping;
  for (NodeDef &node : *item.graph.mutable_node()) {
    node_mapping[node.name()] = &node;
  }

  std::unordered_set<NodeDef *> visited;
  for (NodeDef &node : *item.graph.mutable_node()) {
    // The concat extraction logic is very simple here as we know the number of
    // embedding columns of each benchmark model is > 5, while other concat
    // nodes other than the converging node only have < 5 inputs. Do NOT rely on
    // it to extract converging node in production.
    if (node.op() == "ConcatV2" && node.input_size() > 5) {
      if (!visited.count(&node)) {
        std::stack<NodeDef *> node_stack;
        node_stack.push(&node);
        visited.insert(&node);
        while (!node_stack.empty()) {
          NodeDef *n = node_stack.top();
          n->set_device("/device:CPU:0");
          node_stack.pop();
          for (const std::string &input_tensor : n->input()) {
            const std::string inode_name =
                input_tensor.substr(0, input_tensor.find(":"));
            NodeDef *inode = node_mapping.at(inode_name);
            if (!visited.count(inode)) {
              node_stack.push(inode);
              visited.insert(inode);
            }
          }
        }
      }
    }
  }

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  // opts.expect_device_spec = true;
  auto optimized_graph = std::make_unique<Graph>(OpRegistry::Global());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(opts, item.graph, optimized_graph.get()));
  *options.graph = std::move(optimized_graph);

  return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0, TfCpuGpuPass);

} // namespace feature_opt
} // namespace tensorflow