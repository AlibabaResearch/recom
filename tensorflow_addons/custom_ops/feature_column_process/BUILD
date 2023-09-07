licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "addon_cuda_library")

addon_cuda_library(
    name = "feature_column_process",
    srcs = ["feature_column_process_op_gpu.cu.cc", "feature_column_process_ops.h"],
    deps = []
)