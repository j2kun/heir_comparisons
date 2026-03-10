import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mlp.mlp import CanonicalMLP


def main():
    # 1. Command Line Arguments
    parser = argparse.ArgumentParser(description="Train a CanonicalMLP on MNIST.")
    parser.add_argument(
        "--output_model",
        default="mlp/mlp_model.pth",
        help="Path to save the trained model (e.g., mlp/mlp_model.pth)",
    )
    args = parser.parse_args()

    # 2. Configuration and Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {DEVICE}")

    # 3. Data Preparation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # 4. Initialize Model, Loss, and Optimizer
    model = CanonicalMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}"
                )

        # 6. Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"\nEnd of Epoch {epoch+1}: Test Accuracy: {accuracy:.2f}%\n")

    # 7. Save the Model
    # Ensure the directory exists
    output_dir = os.path.dirname(args.output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Move to CPU before saving to ensure it loads correctly in the conversion script
    torch.save(model.cpu().state_dict(), args.output_model)
    print(f"Model state dictionary saved to {args.output_model}")


if __name__ == "__main__":
    main()
