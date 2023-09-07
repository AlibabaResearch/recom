licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "addon_cc_library")

addon_cc_library(
    name = "shape_construct",
    srcs = ["shape_construct_ops.cc", "shape_construct_ops.h"],
    deps = [
        "//tensorflow_addons:utils",
        "//tensorflow_addons/symbolic_shape:symengine"
    ]
)