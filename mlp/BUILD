load("@hc_pypi//:requirements.bzl", "requirement")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    deps = [
        requirement("torch"),
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":mlp",
        requirement("torch"),
        requirement("torchvision"),
    ],
)

py_binary(
    name = "convert_to_mlir",
    srcs = ["convert_to_mlir.py"],
    data = ["mnist_mlp_model.pth"],
    deps = [
        ":mlp",
        requirement("torch"),
        requirement("torch-mlir"),
    ],
)
