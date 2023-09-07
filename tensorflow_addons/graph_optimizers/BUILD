licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_addons:tensorflow_addons.bzl",
    "custom_op_library",
    "addon_cc_library",
    "addon_cuda_library")

addon_cc_library(
    name = "graph_optimizers",
    deps = [
        "//tensorflow_addons/graph_optimizers:useless_nodes_pruner",
        "//tensorflow_addons/graph_optimizers:shape_construct_optimizer",
        "//tensorflow_addons/graph_optimizers:pre_lookup_optimizer",
        "//tensorflow_addons/graph_optimizers:lookup_optimizer",
        "//tensorflow_addons/graph_optimizers:post_lookup_optimizer",
        "//tensorflow_addons/graph_optimizers:cuda_emitter",
    ]
)

addon_cc_library(
    name = "graph_info",
    srcs = ["graph_info.cc"],
    hdrs = ["graph_info.h"],
    deps = [
        "//tensorflow_addons:utils",
        "//tensorflow_addons/symbolic_shape:op_shape_infer",
        "//tensorflow_addons/symbolic_shape:symbolic_shape_fn",
        "//tensorflow_addons/symbolic_shape:symengine",
    ]
)

addon_cc_library(
    name = "fc_optimizer_base",
    srcs = ["fc_optimizer_base.cc"],
    hdrs = ["fc_optimizer_base.h", "arithm_binary_op_mapping.h"],
    deps = [
        ":graph_info",
        "//tensorflow_addons:utils",
        "//tensorflow_addons/symbolic_shape:op_shape_infer",
        "//tensorflow_addons/symbolic_shape:symbolic_shape_fn",
        "//tensorflow_addons/symbolic_shape:symbolic_shape_fn_registry",
        "//tensorflow_addons/symbolic_shape:symengine",
    ]
)

addon_cc_library(
    name = "useless_nodes_pruner",
    srcs = ["useless_nodes_pruner.cc"],
    hdrs = ["useless_nodes_pruner.h"],
    deps = [
        "//tensorflow_addons:utils",
        ":fc_optimizer_base",
    ]
)

addon_cc_library(
    name = "shape_construct_optimizer",
    srcs = ["shape_construct_optimizer.cc"],
    hdrs = ["shape_construct_optimizer.h"],
    deps = [
        ":fc_optimizer_base",
        "//tensorflow_addons:utils",
        "//tensorflow_addons/symbolic_shape:symbolic_shape_fn",
        "//tensorflow_addons/symbolic_shape:symengine",
    ]
)

addon_cc_library(
    name = "pre_lookup_optimizer",
    srcs = ["pre_lookup_optimizer.cc"],
    hdrs = ["pre_lookup_optimizer.h"],
    deps = [
        ":fc_optimizer_base",
        "@boost//:icl",
        "//tensorflow_addons:utils",
        "//tensorflow_addons/symbolic_shape:symengine",
    ]
)

addon_cc_library(
    name = "lookup_optimizer",
    srcs = ["lookup_optimizer.cc"],
    hdrs = ["lookup_optimizer.h"],
    deps = [
        ":fc_optimizer_base",
        "//tensorflow_addons:utils",
        "//tensorflow_addons/symbolic_shape:symbolic_shape_fn",
        "//tensorflow_addons/symbolic_shape:symengine",
    ]
)

addon_cc_library(
    name = "post_lookup_optimizer",
    srcs = ["post_lookup_optimizer.cc"],
    hdrs = ["post_lookup_optimizer.h"],
    deps = [
        ":fc_optimizer_base",
        "//tensorflow_addons:utils",
        "//tensorflow_addons/symbolic_shape:symbolic_shape_fn",
        "//tensorflow_addons/symbolic_shape:symengine",
    ]
)

addon_cc_library(
    name = "cuda_emitter",
    srcs = ["cuda_emitter.cc"],
    hdrs = ["cuda_emitter.h"],
    deps = [
        ":fc_optimizer_base",
        "//tensorflow_addons:utils",
        "//tensorflow_addons/symbolic_shape:symbolic_shape_fn",
        "//tensorflow_addons/symbolic_shape:symengine",
    ]
)