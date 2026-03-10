import torch
import torch_mlir
import torch_mlir.fx
import argparse
from mlp.mlp import CanonicalMLP


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to MLIR.")

    # Define flagged arguments
    parser.add_argument(
        "--input_model",
        required=True,
        help="Path to the input PyTorch model file (e.g., mlp/mlp_model.pth)",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Path where the resulting .mlir file will be saved (e.g., mlp/mlp.mlir)",
    )

    args = parser.parse_args()

    # Load the model
    model = CanonicalMLP()
    try:
        model.load_state_dict(torch.load(args.input_model))
    except Exception as e:
        print(f"Error loading model from {args.input_model}: {e}")
        return

    model.eval()

    # MNIST input 1x1x28x28
    example_input = torch.randn(1, 1, 28, 28)

    module = torch_mlir.fx.export_and_import(
        model, example_input, output_type=torch_mlir.fx.OutputType.LINALG_ON_TENSORS
    )

    mlir_str = module.operation.get_asm(large_elements_limit=10)

    with open(args.output_file, "w") as f:
        f.write(mlir_str)

    print(f"Successfully converted '{args.input_model}' to '{args.output_file}'")


if __name__ == "__main__":
    main()
