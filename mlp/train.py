import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mlp.mlp import CanonicalMLP


def main():
    # 2. Configuration and Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {DEVICE}")

    # 3. Data Preparation (Load & Transform)
    # Transform: Convert to Tensor and Normalize (Mean 0.1307, Std 0.3081 for MNIST)
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
    criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLLLoss
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}"
                )

        # 6. Quick Evaluation after each epoch
        model.eval()  # Set model to evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"\nEnd of Epoch {epoch+1}: Test Accuracy: {accuracy:.2f}%\n")

    # 7. Emit (Save) the Trained Network
    save_path = "mnist_mlp_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model state dictionary saved to {save_path}")


if __name__ == "__main__":
    main()
