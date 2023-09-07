load("//tensorflow_addons:tensorflow_addons.bzl", "addon_cc_library",)

def op_shape_infer_library_group(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        copts = [],
        **kwargs):
    op_deps = []
    for src in srcs:
        op_name = src.split('.')[0]
        op_deps.append(":" + op_name)
        hdr = op_name + ".h"
        addon_cc_library(
            name = op_name,
            srcs = [src],
            hdrs = [hdr] + hdrs,
            deps = deps,
            copts = copts,
        )

    addon_cc_library(
        name = name,
        deps = op_deps,
        **kwargs,
    )

    # native.filegroup(
    #     name = name,
    #     srcs = op_deps,
    #     **kwargs,
    # )