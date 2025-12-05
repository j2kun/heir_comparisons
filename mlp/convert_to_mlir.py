import torch
import torch_mlir
from mlp.mlp import CanonicalMLP


def main():
    model = CanonicalMLP()
    model.load_state_dict(torch.load("mnist_mlp_model.pth"))

    # Evaluation Mode excludes dropout from the exported model
    model.eval()

    # Tell torch-mlir the shape and type of data the model expects. MNIST input
    # is 1x28x28 (Batch Size 1)
    example_input = torch.randn(1, 1, 28, 28)

    # 'output_type' usually defaults to "linalg_on_tensors" which is
    # the standard entry point for lowering to HEIR.
    module = torch_mlir.compile(
        model, example_input, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
    )

    mlir_str = module.operation.get_asm(large_elements_limit=10)
    with open("mnist_mlp.mlir", "w") as f:
        f.write(mlir_str)
    print("Successfully converted model to 'mlp.mlir'")


if __name__ == "__main__":
    main()
