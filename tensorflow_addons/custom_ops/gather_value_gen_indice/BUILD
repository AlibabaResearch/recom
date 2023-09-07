licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "addon_cc_library")

addon_cc_library(
    name = "gather_value_gen_indice",
    srcs = ["gather_value_gen_indice_ops.cc", "gather_value_gen_indice_ops.h"],
    deps = [
        "//tensorflow_addons/symbolic_shape:symengine"
    ]
)