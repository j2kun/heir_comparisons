// A shim around the HEIR OpenFHE interpreter to provide pybind11 bindings.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlp_interpreter_shim.h"

namespace py = pybind11;


PYBIND11_MODULE(mlp_interpreter, m) {
  m.def("mnist_interpreter", &mnist_interpreter, py::call_guard<py::gil_scoped_release>());
}
