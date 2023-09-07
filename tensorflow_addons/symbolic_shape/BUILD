licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load("//tensorflow_addons/symbolic_shape:symbolic_shape.bzl", "op_shape_infer_library_group")
load(
    "//tensorflow_addons:tensorflow_addons.bzl",
    "custom_op_library",
    "addon_cc_library",
    "addon_cuda_library")

addon_cc_library(
    name = "symbolic_shape",
    deps = [
        ":symengine",
        ":symbolic_shape_fn",
        ":symbolic_shape_fn_registry",
        ":shape_infer_utils",
        ":op_shape_infer",
    ]
)

cmake(
    name = "symengine",
    # set thread number accroding to your available cores
    build_args = ["-j64"],
    targets = [
        "symengine",
    ],
    lib_source = "@symengine//:all_srcs",
    out_static_libs = ["libsymengine.a"],
)

addon_cc_library(
    name = "symbolic_shape_fn",
    srcs = ["symbolic_shape_fn.cc"],
    hdrs = ["symbolic_shape_fn.h"],
    deps = [":symengine", "//tensorflow_addons:utils"]
)

addon_cc_library(
    name = "symbolic_shape_fn_registry",
    srcs = ["symbolic_shape_fn_registry.cc"],
    hdrs = ["symbolic_shape_fn_registry.h"],
    deps = [
        ":symengine",
        "//tensorflow_addons:utils",
        ":symbolic_shape_fn"
    ]
)

addon_cc_library(
    name = "shape_infer_utils",
    srcs = ["shape_infer_utils.cc"],
    hdrs = ["shape_infer_utils.h"],
    deps = [
        ":symengine",
        "//tensorflow_addons:utils",
        ":symbolic_shape_fn",
        ":symbolic_shape_fn_registry"
    ]
)

op_shape_infer_library_group(
    name = "op_shape_infer",
    srcs = glob(["op_infer_fn/*.cc"]),
    deps = [
        ":symengine",
        "//tensorflow_addons:utils",
        ":symbolic_shape_fn",
        ":symbolic_shape_fn_registry",
        ":shape_infer_utils"
    ]
)
