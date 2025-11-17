from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common.config import load_config, set_seed, parse_args, get_device
from src.models.lenet import LeNet


def get_dataloaders(config: dict):
    data_cfg = config["data"]
    root = data_cfg["root"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)

    transform = transforms.Compose(
        [
            transforms.Pad(2),  # 28x28 -> 32x32 for LeNet
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    device = get_device(config.get("device", "cuda"))
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(config)

    model_cfg = config["model"]
    model = LeNet(num_classes=model_cfg.get("num_classes", 10)).to(device)

    train_cfg = config["train"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])
    epochs = train_cfg["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

    out_cfg = config["outputs"]
    artifacts_dir = Path(out_cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / out_cfg["model_filename"]

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
