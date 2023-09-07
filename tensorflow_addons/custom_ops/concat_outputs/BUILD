licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "addon_cuda_library")

addon_cuda_library(
    name = "concat_outputs",
    srcs = ["concat_outputs_op_gpu.cu.cc", "concat_outputs_ops.h"],
    deps = [
        "//tensorflow_addons:utils"
    ]
)