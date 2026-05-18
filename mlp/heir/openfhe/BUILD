load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_heir//heir:openfhe.bzl", "heir_openfhe_lib")
load("@rules_python//python:py_test.bzl", "py_test")

heir_openfhe_lib(
    name = "mlp",
    cc_lib_target_name = "mlp_cc_lib",
    generated_lib_header = "mlp.inc.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=bgv",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-openfhe",
    ],
    mlir_src = "mlp.mlir",
    pybind_target_name = "mlp_pybind",
)

py_test(
    name = "mlp_test",
    size = "enormous",
    srcs = ["mlp_test.py"],
    data = [
        "//mlp/data:t10k-images-idx3-ubyte",
        "//mlp/data:t10k-labels-idx1-ubyte",
        "//mlp/data:traced_model.pt",
    ],
    main = "mlp_test.py",
    tags = [
        "requires-mem:28g",
    ],
    deps = [
        ":mlp_pybind",
        "@abseil-py//absl/testing:absltest",
        "@hc_pypi//numpy",
        "@hc_pypi//torch",
    ],
)
