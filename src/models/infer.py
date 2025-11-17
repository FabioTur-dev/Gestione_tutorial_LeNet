from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.common.config import load_config, set_seed, get_device
from src.models.lenet import LeNet


MEAN = 0.1307
STD = 0.3081


def build_test_loader(config: dict, batch_size: int = 1, shuffle: bool = True):
    data_cfg = config["data"]
    root = data_cfg["root"]
    num_workers = 0  # meglio 0 per interattivo su Windows

    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,)),
        ]
    )

    test_dataset = datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


def denormalize(img_tensor):
    """Inverti la normalizzazione per visualizzare l'immagine (1x32x32)."""
    img = img_tensor.clone().detach().cpu()
    img = img * STD + MEAN
    img = img.clamp(0.0, 1.0)
    return img


def interactive_inference(config_path: str, seed: int | None, num_samples: int):
    # carica config e setta seed/device
    config = load_config(config_path)
    if seed is None:
        seed = config.get("seed", 42)
    set_seed(seed)

    device = get_device(config.get("device", "cuda"))
    print(f"Using device: {device}")

    # dataloader
    loader = build_test_loader(config, batch_size=1, shuffle=True)

    # modello
    model_cfg = config["model"]
    model = LeNet(num_classes=model_cfg.get("num_classes", 10)).to(device)

    out_cfg = config["outputs"]
    artifacts_dir = Path(out_cfg["artifacts_dir"])
    model_path = artifacts_dir / out_cfg["model_filename"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\nEntering interactive inference mode.")
    print("Press ENTER for next image, 'q' + ENTER to quit.\n")

    shown = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(dim=1)

        img_vis = denormalize(images[0])
        img_vis = img_vis.squeeze(0)  # 1x32x32 -> 32x32

        plt.imshow(img_vis.numpy(), cmap="gray")
        plt.title(f"True: {labels.item()} | Pred: {preds.item()}")
        plt.axis("off")
        plt.show()

        shown += 1
        if shown >= num_samples:
            print("Reached requested number of samples, exiting.")
            break

        user = input("Next image? [ENTER=continue, q=quit] ")
        if user.strip().lower().startswith("q"):
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive MNIST inference with LeNet"
    )
    parser.add_argument(
        "--config",
        default="configs/dev.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config if provided)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test images to show",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    interactive_inference(args.config, args.seed, args.num_samples)


if __name__ == "__main__":
    main()
