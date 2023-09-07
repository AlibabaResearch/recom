load("@local_config_tf//:build_defs.bzl", "D_GLIBCXX_USE_CXX11_ABI")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")


def addon_cc_library(
        deps = [],
        copts = [],
        **kwargs):
    deps = deps + [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ]

    copts = copts + select({
        "//conditions:default": ["-pthread", "-std=c++14", D_GLIBCXX_USE_CXX11_ABI],
    })

    native.cc_library(
        copts = copts,
        alwayslink = 1,
        features = select({
            "//conditions:default": [],
        }),
        deps = deps,
        **kwargs
    )


def addon_cuda_library(
        deps = [],
        copts = [],
        **kwargs):
    deps = deps + [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ]

    copts = copts + if_cuda(["-DGOOGLE_CUDA=1"]) + if_cuda_is_configured([
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
    ])
    deps = deps + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ])

    native.cc_library(
        deps = deps,
        copts = copts,
        alwayslink = 1,
        **kwargs
    )


def custom_op_library(
        name,
        srcs = [],
        cuda_srcs = [],
        deps = [],
        cuda_deps = [],
        copts = [],
        **kwargs):
    deps = deps + [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ]

    if cuda_srcs:
        copts = copts + if_cuda(["-DGOOGLE_CUDA=1"])
        cuda_copts = copts + if_cuda_is_configured([
            "-x cuda",
            "-nvcc_options=relaxed-constexpr",
            "-nvcc_options=ftz=true",
        ])
        cuda_deps = deps + if_cuda_is_configured(cuda_deps) + if_cuda_is_configured([
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart_static",
        ])
        basename = name.split(".")[0]
        native.cc_library(
            name = basename + "_gpu",
            srcs = cuda_srcs,
            deps = cuda_deps,
            copts = cuda_copts,
            alwayslink = 1,
            **kwargs
        )
        deps = deps + if_cuda_is_configured([":" + basename + "_gpu"])

    copts = copts + select({
        "//conditions:default": ["-pthread", "-std=c++14", D_GLIBCXX_USE_CXX11_ABI],
    })

    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = copts,
        linkshared = 1,
        features = select({
            "//conditions:default": [],
        }),
        deps = deps,
        **kwargs
    )
