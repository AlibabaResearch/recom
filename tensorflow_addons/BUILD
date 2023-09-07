licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow_addons:tensorflow_addons.bzl",
    "custom_op_library",
    "addon_cc_library",
    "addon_cuda_library")

custom_op_library(
    name = "librecom.so",
    deps = [
        "//tensorflow_addons/passes:fc_optimize_pass"
    ],
    linkopts = [
        "-lgmp",
        "-ldl",
    ],
)

addon_cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
)

custom_op_library(
    name = "libtf_cpu_gpu.so",
    deps = [
        "//tensorflow_addons/passes:tf_cpu_gpu_pass"
    ],
)

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = {
        "//tensorflow_addons:librecom.so": "--cxxopt=-xc++",
    },
    # No need to add flags already in .bazelrc. They're automatically picked up.
    # If you don't need flags, a list of targets is also okay, as is a single target string.
    # Wildcard patterns, like //... for everything, *are* allowed here, just like a build.
      # As are additional targets (+) and subtractions (-), like in bazel query https://docs.bazel.build/versions/main/query.html#expressions
    # And if you're working on a header-only library, specify a test or binary target that compiles it.
)