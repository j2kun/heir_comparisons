// A wrapper around an invocation of the C++ interpreter for OpenFHE customized
// for the mnist model test. This is a workaorund due to long compilation times
// of generated C++ functions with >> 50k lines.

#include <string>
#include <utility>
#include <vector>

/// Run the mnist model on the given input data, returning the output vector
/// and the time taken in seconds.
///
/// @param mlirSrc The MLIR OpenFHE dialect IR to run, as a string
/// @param inputData The input vector to pass to the model.
std::pair<std::vector<float>, double> mnist_interpreter(
    const std::string& mlirSrc,
    const std::vector<float>& inputData);
