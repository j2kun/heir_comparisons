"""Build settings for OpenFHE and OpenMP."""

OPENFHE_VERSION_MAJOR = 1

OPENFHE_VERSION_MINOR = 11

OPENFHE_VERSION_PATCH = 3

OPENFHE_VERSION = "{}.{}.{}".format(OPENFHE_VERSION_MAJOR, OPENFHE_VERSION_MINOR, OPENFHE_VERSION_PATCH)

OPENFHE_DEFINES = [
    "MATHBACKEND=2",
    "OPENFHE_VERSION=" + OPENFHE_VERSION,
    "PARALLEL",
]

OPENFHE_COPTS = [
    "-Wno-non-virtual-dtor",
    "-Wno-shift-op-parentheses",
    "-Wno-unused-private-field",
    "-fexceptions",
]

OPENFHE_LINKOPTS = [
    "-fopenmp",
    "-lomp",
]

OPENMP_COPTS = [
    "-fopenmp",
    "-Xpreprocessor",
    "-Wno-unused-command-line-argument",
]
