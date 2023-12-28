load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_point_pillars",
    srcs = [
        "src/pillarScatter.cpp",
        #"src/pillarScatterKernels.cu",
        "src/pointpillar.cpp",
        #"src/postprocess_kernels.cu",
        "src/postprocess.cpp",
        #"src/preprocess_kernels.cu",
        "src/preprocess.cpp",
    ],
    hdrs = [
        "include/kernel.h",
        "include/params.h",
        "include/pillarScatter.h",
        "include/pointpillar.h",
        "include/postprocess.h",
        "include/preprocess.h",
    ],
    copts = [
        '-DMODULE_NAME=\\"cuda_point_pillars\\"',
    ],
    deps = [
        "@local_config_cuda//cuda:cudart",
        "@local_config_tensorrt//:tensorrt",
    ],
)

cpplint()
