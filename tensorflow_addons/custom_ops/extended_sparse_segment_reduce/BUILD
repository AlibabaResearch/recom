licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "addon_cc_library")

addon_cc_library(
    name = "extended_sparse_segment_reduce",
    srcs = ["extended_sparse_segment_reduce_ops.cc", "extended_sparse_segment_reduce_ops.h"],
    deps = [
        "//tensorflow_addons/symbolic_shape:symengine"
    ]
)