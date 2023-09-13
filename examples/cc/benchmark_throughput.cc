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

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_replace.h"
#include "cxxopts.hpp"
#include "tensorflow/core/public/session.h"
#include "utils.h"

using namespace ::tensorflow;
using namespace ::tensorflow::benchmark;

int main(int argc, char **argv) {
  cxxopts::Options options(
      "benchmark_throughput",
      "Using pressure test to evaluate the throughput of TF recommendation "
      "models with mutliple inference threads");

  // clang-format off
  options.add_options()
    ("model_path", "Path of the TF saved model (mandatory)", cxxopts::value<std::string>())
    ("serve_workers", "Number of workers serving the requests concurrently", cxxopts::value<int>()->default_value("1"))
    ("sla_latency", "Service level agreemnet latency", cxxopts::value<float>()->default_value("100"))
    ("intra_threads", "TF intra_threads_parallelism (CPU)", cxxopts::value<int>()->default_value("0"))
    ("inter_threads", "TF inter_threads_parallelism (CPU)", cxxopts::value<int>()->default_value("0"))
    ("num_iterations", "Number of running iteration", cxxopts::value<int>()->default_value("100"))
    ("lib_path", "Link a TF addon library", cxxopts::value<std::string>())
    ("disable_gpu", "Disable the Usage of GPU", cxxopts::value<bool>()->default_value("false"))
    ("embedding_only", "Only process the embedding parts", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Show usage");
  // clang-format on

  auto args = options.parse(argc, argv);
  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  if (!args.count("model_path")) {
    std::cerr << "ERROR: model_path must be specified!" << std::endl;
    std::cerr << options.help() << std::endl;
    exit(-1);
  }

  const auto &model_path = args["model_path"].as<std::string>();
  const auto &serve_workers = args["serve_workers"].as<int>();
  const auto &sla_latency = args["sla_latency"].as<float>();
  const auto &intra_threads = args["intra_threads"].as<int>();
  const auto &inter_threads = args["inter_threads"].as<int>();
  const auto &num_iterations = args["num_iterations"].as<int>();
  const bool &use_gpu = !args["disable_gpu"].as<bool>();

  const bool &use_lib = args.count("lib_path");
  const std::string &lib_path =
      use_lib ? args["lib_path"].as<std::string>() : "";

  LOG(INFO) << "Testing with model: " << model_path;
  LOG(INFO) << "Testing with serve_workers: " << serve_workers;
  LOG(INFO) << "Testing with SLA latency: " << sla_latency;
  LOG(INFO) << "Testing with intra_threads: " << intra_threads;
  LOG(INFO) << "Testing with inter_threads: " << inter_threads;
  LOG(INFO) << "Testing with num_iterations: " << num_iterations;
  LOG(INFO) << "GPU is " << (use_gpu ? "" : "NOT ") << "enabled";

  if (use_lib) {
    LOG(INFO) << "Testing with library: " << lib_path;
    LoadOpLibrary(lib_path);
  }

  std::shared_ptr<SavedModelBundle> bundle;
  LoadSavedModelWrap(model_path, intra_threads, inter_threads, bundle, use_gpu);
  const auto &gd = bundle->meta_graph_def.graph_def();

  std::vector<std::string> outputs;
  if (!args["embedding_only"].as<bool>()) {
    ExtractGraphOutputs(bundle, outputs);
  } else {
    ExtractGraphConcats(gd, outputs);
  }

  auto RunPredict =
      [&](const std::vector<std::pair<std::string, Tensor>> &inputs,
          std::vector<Tensor> &results) {
        Status status = bundle->session->Run(inputs, outputs, {}, &results);
        if (!status.ok()) {
          LOG(ERROR) << "Failed to predict! status = " << status.ToString();
          exit(-1);
        }
      };

  auto RunMultiThreads =
      [&](const std::vector<std::pair<std::string, Tensor>> &inputs,
          std::vector<Tensor> &results,
          int batch_size) -> std::pair<float, int> {
    // LOG(INFO) << "Fetch nodes: " << absl::StrJoin(outputs, ", ");

    using milli = std::chrono::milliseconds;

    LOG(INFO) << "Start warmup predict";

    auto t1 = std::chrono::high_resolution_clock::now();
    RunPredict(inputs, results);
    auto t2 = std::chrono::high_resolution_clock::now();

    int warmup_time = std::chrono::duration_cast<milli>(t2 - t1).count();
    LOG(INFO) << "Warmup done. Time: " << warmup_time << "ms";

    auto run_thread = [&](int idx, int total) {
      for (int i = 0; i < num_iterations; ++i) {
        std::vector<Tensor> res;
        RunPredict(inputs, res);
      }
    };

    auto t3 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < serve_workers; i++) {
      threads.push_back(std::thread(run_thread, i, serve_workers));
    }
    for (auto &thread : threads) {
      thread.join();
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    int total_latency = std::chrono::duration_cast<milli>(t4 - t3).count();
    float avg_latency = static_cast<float>(total_latency) / num_iterations;
    int throughput = serve_workers * num_iterations * batch_size /
                     static_cast<float>(total_latency) * 1000;

    return {avg_latency, throughput};
  };

  int max_throughput = 0;
  int max_batch_size = 0;
  int batch_size = 16;
  while (true) {
    std::vector<std::pair<std::string, Tensor>> inputs;

    ConstructGraphInputs(gd, batch_size, inputs);

    std::vector<Tensor> results;
    auto latency_throughput = RunMultiThreads(inputs, results, batch_size);

    float latency = latency_throughput.first;
    int throughput = latency_throughput.second;
    LOG(INFO) << "batch size: " << batch_size << ", "
              << "avg latency: " << latency << ", "
              << "throughput: " << throughput;

    if (latency >= sla_latency) {
      if (batch_size > 512) {
        break;
      } else {
        batch_size += 50;
      }
    } else {
      max_batch_size = batch_size;
      max_throughput = throughput;

      int increment;
      if (latency < 50) {
        increment = batch_size;
      } else if (latency < 70) {
        increment = batch_size / 5;
      } else if (latency < 80) {
        increment = batch_size / 10;
      } else if (latency < 90) {
        increment = batch_size / 30;
      } else {
        increment = batch_size / 100;
      }
      batch_size += std::max(5, increment);
    }
  }

  LOG(INFO) << "[Throughput Result] max_batch_size " << max_batch_size
            << ", max_throughput " << max_throughput;
}