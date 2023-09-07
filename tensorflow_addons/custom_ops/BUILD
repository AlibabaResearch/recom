licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_addons:tensorflow_addons.bzl", "addon_cc_library")

addon_cc_library(
    name = "custom_ops",
    deps = [
        "//tensorflow_addons/custom_ops/concat_inputs",
        "//tensorflow_addons/custom_ops/concat_outputs",
        "//tensorflow_addons/custom_ops/extended_sparse_segment_reduce",
        "//tensorflow_addons/custom_ops/extended_sparse_to_dense",
        "//tensorflow_addons/custom_ops/feature_column_process",
        "//tensorflow_addons/custom_ops/gather_indice_value",
        "//tensorflow_addons/custom_ops/gather_str_value_gen_indice",
        "//tensorflow_addons/custom_ops/gather_value_gen_indice",
        "//tensorflow_addons/custom_ops/select_value",
        "//tensorflow_addons/custom_ops/shape_construct",
    ]
)
