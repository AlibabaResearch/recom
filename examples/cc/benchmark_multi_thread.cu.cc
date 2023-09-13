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

#include <cuda_profiler_api.h>
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
  cxxopts::Options options("benchmark_multi_thread",
                           "Benchmark the performance of TF recommendation "
                           "models with mutliple inference threads");

  // clang-format off
  options.add_options()
    ("model_path", "Path of the TF saved model (mandatory)", cxxopts::value<std::string>())
    ("batch_size", "Inference batch size", cxxopts::value<int>()->default_value("256"))
    ("serve_workers", "Number of workers serving the requests concurrently", cxxopts::value<int>()->default_value("1"))
    ("intra_threads", "TF intra_threads_parallelism (CPU)", cxxopts::value<int>()->default_value("0"))
    ("inter_threads", "TF inter_threads_parallelism (CPU)", cxxopts::value<int>()->default_value("0"))
    ("num_iterations", "Number of running iteration", cxxopts::value<int>()->default_value("100"))
    ("lib_path", "Link a TF addon library", cxxopts::value<std::string>())
    ("disable_gpu", "Disable the Usage of GPU", cxxopts::value<bool>()->default_value("false"))
    ("embedding_only", "Only process the embedding parts", cxxopts::value<bool>()->default_value("false"))
    ("cuda_profile", "Profiling the kernels (single thread). Only works when GPU is used", cxxopts::value<bool>()->default_value("false"))
    ("timeline", "Record the TF timeline (single thread) to the specified path", cxxopts::value<std::string>())
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
  const auto &batch_size = args["batch_size"].as<int>();
  const auto &serve_workers = args["serve_workers"].as<int>();
  const auto &intra_threads = args["intra_threads"].as<int>();
  const auto &inter_threads = args["inter_threads"].as<int>();
  const auto &num_iterations = args["num_iterations"].as<int>();
  const bool &use_gpu = !args["disable_gpu"].as<bool>();

  const bool &use_lib = args.count("lib_path");
  const std::string &lib_path =
      use_lib ? args["lib_path"].as<std::string>() : "";

  LOG(INFO) << "Testing with model: " << model_path;
  LOG(INFO) << "Testing with batch_size: " << batch_size;
  LOG(INFO) << "Testing with serve_workers: " << serve_workers;
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
          std::vector<Tensor> &results, int batch_size) {
        // LOG(INFO) << "Fetch nodes: " << absl::StrJoin(outputs, ", ");

        using milli = std::chrono::milliseconds;

        LOG(INFO) << "Start warmup predict";

        auto t1 = std::chrono::high_resolution_clock::now();
        RunPredict(inputs, results);
        auto t2 = std::chrono::high_resolution_clock::now();

        int warmup_time = std::chrono::duration_cast<milli>(t2 - t1).count();
        LOG(INFO) << "Warmup done. Time: " << warmup_time << "ms";

        auto run_thread = [&](int idx, int total) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i = 0; i < num_iterations; ++i) {
            std::vector<Tensor> res;
            RunPredict(inputs, res);
          }
          auto t2 = std::chrono::high_resolution_clock::now();

          LOG(INFO) << "Batch size " << batch_size << ", sess run "
                    << num_iterations << " times, average latency: "
                    << static_cast<float>(
                           std::chrono::duration_cast<milli>(t2 - t1).count()) /
                           num_iterations
                    << " ms";
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
        LOG(INFO) << "Total latency: " << total_latency << "ms";

        int throughput = serve_workers * num_iterations * batch_size /
                         static_cast<float>(total_latency) * 1000;
        LOG(INFO) << "Batch size: " << batch_size
                  << ", Throughput: " << throughput << " inference/s";
      };

  std::vector<std::pair<std::string, Tensor>> inputs;
  ConstructGraphInputs(gd, batch_size, inputs);

  std::vector<Tensor> results;
  RunMultiThreads(inputs, results, batch_size);

  if (args.count("cuda_profile")) {
    for (int i = 0; i < 10; ++i)
      RunPredict(inputs, results);

    cudaProfilerStart();
    RunPredict(inputs, results);
    cudaProfilerStop();
  }

  if (args.count("timeline")) {
    for (int i = 0; i < 10; ++i)
      RunPredict(inputs, results);

    RunOptions run_options;
    run_options.set_trace_level(RunOptions::FULL_TRACE);
    RunMetadata run_metadata;
    Status status = bundle->session->Run(run_options, inputs, outputs, {},
                                         &results, &run_metadata);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to predict! status = " << status.ToString();
      exit(-1);
    }

    std::string tl_str;
    run_metadata.step_stats().SerializeToString(&tl_str);
    std::ofstream ofs(args["timeline"].as<std::string>());
    ofs << tl_str;
    ofs.close();
  }
}
