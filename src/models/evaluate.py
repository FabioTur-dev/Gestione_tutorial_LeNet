from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common.config import load_config, set_seed, parse_args, get_device
from src.models.lenet import LeNet


def get_test_loader(config: dict):
    data_cfg = config["data"]
    root = data_cfg["root"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)

    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_dataset = datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_loader


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

    test_loader = get_test_loader(config)

    model_cfg = config["model"]
    model = LeNet(num_classes=model_cfg.get("num_classes", 10)).to(device)

    out_cfg = config["outputs"]
    artifacts_dir = Path(out_cfg["artifacts_dir"])
    model_path = artifacts_dir / out_cfg["model_filename"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")

    reports_dir = Path(out_cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / out_cfg["metrics_filename"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"accuracy: {accuracy:.4f}\n")

    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
