load("//bazel/openfhe:copts.bzl", "OPENFHE_LINKOPTS", "OPENMP_COPTS")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_heir//heir:heir_opt.bzl", "heir_opt")
load("@rules_heir//heir:heir_translate.bzl", "heir_translate")
load("@rules_python//python:py_test.bzl", "py_test")

# Optimize and generate HEIR's OpenFHE dialect IR
heir_opt(
    name = "mlp_heir_opt",
    src = "//mlp:mlp.mlir",
    generated_filename = "mlp.openfhe.mlir",
    passes = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-openfhe",
    ],
)

# Codegen for OpenFHE C++ code
heir_translate(
    name = "mlp_translate_cc",
    src = "mlp.openfhe.mlir",
    generated_filename = "mlp.inc.cc",
    passes = [
        "--emit-openfhe-pke",
        "--openfhe-include-type=source-relative",
    ],
)

# Codegen for OpenFHE C++ header
heir_translate(
    name = "mlp_translate_h",
    src = "mlp.openfhe.mlir",
    generated_filename = "mlp.inc.h",
    passes = [
        "--emit-openfhe-pke-header",
        "--openfhe-include-type=source-relative",
    ],
)

# cc_library to bundle the generated C++ code
cc_library(
    name = "mlp_compiled_cc_lib",
    srcs = ["mlp.inc.cc"],
    hdrs = ["mlp.inc.h"],
    deps = ["@openfhe//:pke"],
    copts = OPENMP_COPTS + ["-O0"],
    linkopts = OPENFHE_LINKOPTS,
)

# Codegen for OpenFHE C++ pybindings
heir_translate(
    name = "mlp_translate_pybind",
    src = "mlp.openfhe.mlir",
    passes = [
        "--openfhe-include-type=source-relative",
        "--emit-openfhe-pke-pybind",
        "--pybind-header-include=mlp.inc.h",
        # The module name here needs to match the target name,
        # as this is what pybind_extension uses for the name
        # exposed to `import`
        "--pybind-module-name=mlp_translate_pybind",
    ],
    generated_filename = "mlp_openfhe_bindings.cpp",
)

# Build the python extension module
pybind_extension(
    name = "mlp_pybind",
    srcs = ["mlp_openfhe_bindings.cpp"],
    deps = [
        ":mlp_translate_pybind",
        ":mlp_compiled_cc_lib",
        "@openfhe//:pke",
    ],
)

# HEIR OpenFHE interpreter shim, so that python code can
# invoke the interpreted mode instead of the compiled mode.
cc_library(
    name = "interpreter_shim",
    srcs = ["interpreter_shim.cpp"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@openfhe//:pke",
        "@heir//lib/Target/OpenFhePke:Interpreter",
    ],
)

# Test and time it
py_test(
    name = "mlp_compiled_test",
    size = "enormous",
    srcs = ["mlp_compiled_test.py"],
    main = "mlp_compiled_test.py",
    tags = [
        "requires-mem:28g",
    ],
    deps = [
        ":mlp_pybind",
        "@abseil-py//absl/testing:absltest",
        "@hc_pypi//numpy",
        "@hc_pypi//torch",
    ],
    data = [
        "//mlp/data:t10k-images-idx3-ubyte",
        "//mlp/data:t10k-labels-idx1-ubyte",
        "//mlp/data:traced_model.pt",
    ],
)
