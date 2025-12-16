load("//bazel/openfhe:copts.bzl", "OPENFHE_LINKOPTS", "OPENMP_COPTS")
load("@hc_pypi//:requirements.bzl", "requirement") load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_heir//heir:heir_opt.bzl", "heir_opt")
load("@rules_heir//heir:heir_translate.bzl", "heir_translate")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

# Model specification
py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    deps = [
        requirement("torch"),
    ],
)

# Training script
py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":mlp",
        requirement("torch"),
        requirement("torchvision"),
    ],
)

# Export from PyTorch to MLIR
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

# Optimize and generate HEIR's OpenFHE dialect IR
heir_opt(
    name = "mnist_mlp_openfhe_heir",
    src = "mnist_mlp.mlir",
    generated_filename = "mnist_mlp.openfhe.mlir",
    passes = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-openfhe",
    ],
)

# Codegen for OpenFHE C++ code
heir_translate(
    name = "mnist_mlp_translate_cc",
    src = ":mnist_mlp.openfhe.mlir",
    generated_filename = "mnist_mlp.inc.cc",
    passes = [
        "--emit-openfhe-pke",
        "--openfhe-include-type=source-relative",
    ],
)

# Codegen for OpenFHE C++ header
heir_translate(
    name = "mnist_mlp_translate_h",
    src = ":mnist_mlp.openfhe.mlir",
    generated_filename = "mnist_mlp.inc.h",
    passes = [
        "--emit-openfhe-pke-header",
        "--openfhe-include-type=source-relative",
    ],
)

# cc_library to bundle the generated C++ code
cc_library(
    name = "mnist_mlp_heir_openfhe_cc_lib",
    srcs = ["mnist_mlp.inc.cc"],
    hdrs = ["mnist_mlp.inc.h"],
    deps = ["@openfhe//:pke"],
    copts = OPENMP_COPTS,
    linkopts = OPENFHE_LINKOPTS,
)

# Codegen for OpenFHE C++ pybindings
heir_translate(
    name = "mnist_mlp_openfhe.heir_translate_pybind",
    src = "mnist_mlp.openfhe.mlir",
    passes = [
        "--openfhe-include-type=source-relative",
        "--emit-openfhe-pke-pybind",
        "--pybind-header-include=mnist_mlp.inc.h",
        # The module name here needs to match the target name,
        # as this is what pybind_extension uses for the name
        # exposed to `import`
        "--pybind-module-name=mnist_mlp_translate_pybind",
    ],
    generated_filename = "mnist_mlp_openfhe_bindings.cpp",
)

# Build the python extension module
pybind_extension(
    name = "mnist_mlp_openfhe_pybind",
    srcs = ["mnist_mlp_openfhe_bindings.cpp"],
    deps = [
        ":mnist_mlp_openfhe.heir_translate_pybind",
        ":mnist_mlp_heir_openfhe_cc_lib",
        "@openfhe//:pke",
    ],
)

# Test and time it
py_test(
    name = "mnist_mlp_heir_test",
    size = "enormous",
    srcs = ["mnist_mlp_heir_test.py"],
    main = "mnist_mlp_heir_test.py",
    tags = [
        "requires-mem:28g",
    ],
    deps = [
        ":mnist_mlp_openfhe_pybind",
        "@abseil-py//absl/testing:absltest",
        "@hc_pypi//numpy",
        "@hc_pypi//torch",
    ],
    data = [
        ":mnist_mlp.openfhe.mlir",
        "//mlp/data:t10k-images-idx3-ubyte",
        "//mlp/data:t10k-labels-idx1-ubyte",
        "//mlp/data:traced_model.pt",
    ],
)
