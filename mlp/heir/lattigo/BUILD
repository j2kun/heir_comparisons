load("@rules_go//go:def.bzl", "go_test")
load("@rules_heir//heir:lattigo.bzl", "heir_lattigo_lib")

heir_lattigo_lib(
    name = "mlp",
    go_library_name = "mlp",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=ckks",
        "--torch-linalg-to-ckks=ciphertext-degree=8192",
        "--scheme-to-lattigo",
    ],
    mlir_src = "//mlp:mlp.mlir",
)

go_test(
    name = "mlp_test",
    size = "large",
    srcs = ["mlp_test.go"],
    data = [
        "//mlp/data:t10k-images-idx3-ubyte",
        "//mlp/data:t10k-labels-idx1-ubyte",
        "//mlp/data:traced_model.pt",
    ],
    embed = [":mlp"],
)
