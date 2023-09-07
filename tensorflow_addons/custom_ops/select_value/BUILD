licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "addon_cc_library")

addon_cc_library(
    name = "select_value",
    srcs = ["select_value_ops.cc", "select_value_ops.h"],
    deps = [
        "//tensorflow_addons/symbolic_shape:symengine"
    ]
)