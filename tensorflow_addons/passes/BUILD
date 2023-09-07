licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_addons:tensorflow_addons.bzl",
    "custom_op_library",
    "addon_cc_library",
    "addon_cuda_library")

addon_cc_library(
    name = "fc_optimize_pass",
    srcs = ["fc_optimize_pass.cc"],
    hdrs = ["fc_optimize_pass.h"],
    deps = [
        "//tensorflow_addons:utils",
        "//tensorflow_addons/graph_optimizers:useless_nodes_pruner",
        "//tensorflow_addons/graph_optimizers:shape_construct_optimizer",
        "//tensorflow_addons/graph_optimizers:pre_lookup_optimizer",
        "//tensorflow_addons/graph_optimizers:lookup_optimizer",
        "//tensorflow_addons/graph_optimizers:post_lookup_optimizer",
        "//tensorflow_addons/graph_optimizers:cuda_emitter",
        "//tensorflow_addons/custom_ops/concat_inputs",
        "//tensorflow_addons/custom_ops/concat_outputs",
        "//tensorflow_addons/custom_ops/feature_column_process",
        "//tensorflow_addons/custom_ops/shape_construct",
        "//tensorflow_addons/custom_ops/extended_sparse_to_dense",
        "//tensorflow_addons/custom_ops/gather_str_value_gen_indice",
    ]
)

addon_cc_library(
    name = "tf_cpu_gpu_pass",
    srcs = ["tf_cpu_gpu_pass.cc"]
)