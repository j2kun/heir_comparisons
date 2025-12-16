#include <string>
#include <utility>
#include <vector>

#include "lib/Target/OpenFhePke/Interpreter.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/include/mlir/Parser/Parser.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project
#include "src/pke/include/openfhe.h"           // from @openfhe

using namespace lbcrypto;
using namespace mlir::heir::openfhe;
using CryptoContextT = CryptoContext<DCRTPoly>;

std::pair<std::vector<float>, double>
mnist_interpreter(const std::string &mlirSrc,
                  const std::vector<float> &input) {
  // Load the MLIR module from a file
  MLIRContext context;
  initContext(context);
  OwningOpRef<ModuleOp> module = parse(&context, mlirSrc);
  Interpreter interpreter(module.get());

  auto start = std::chrono::high_resolution_clock::now();
  TypedCppValue ccInitial =
      interpreter.interpret("mlp__generate_crypto_context", {})[0];

  auto keyPair = std::get<CryptoContextT>(ccInitial.value)->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  std::vector<TypedCppValue> args = {ccInitial, TypedCppValue(secretKey)};
  TypedCppValue cc = std::move(
      interpreter.interpret("mlp__configure_crypto_context", args)[0]);

  TypedCppValue arg0Enc = interpreter.interpret(
      "mlp__encrypt__arg0",
      {cc, TypedCppValue(input), TypedCppValue(publicKey)})[0];

  TypedCppValue outputEncrypted =
      interpreter.interpret("mlp", {cc, arg0Enc})[0];

  TypedCppValue outputDecrypted =
      interpreter.interpret("mlp__decrypt__output",
                            {cc, outputEncrypted, TypedCppValue(secretKey)})[0];

  auto end = std::chrono::high_resolution_clock::now();
  double duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  return {std::get<std::vector<float>>(outputDecrypted.value), duration};
}
