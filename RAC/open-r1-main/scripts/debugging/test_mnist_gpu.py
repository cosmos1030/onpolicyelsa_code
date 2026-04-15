#!/usr/bin/env python
"""Small GPU smoke test with MNIST training.

This is intentionally lightweight so we can separate cluster/GPU health from
the larger Open-R1 training stack. If MNIST download is unavailable on compute
nodes, the script falls back to synthetic image data and still exercises CUDA.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def run_cmd(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    try:
        out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("command not found")
        return

    if out.stdout:
        print(out.stdout.strip())
    if out.stderr:
        print(out.stderr.strip(), file=sys.stderr)


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def print_cuda_status(label: str, device: torch.device) -> None:
    if device.type != "cuda":
        print(f"[{label}] running on CPU")
        return

    free, total = torch.cuda.mem_get_info(device.index or 0)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(
        f"[{label}] free={bytes_to_gib(free):.2f} GiB "
        f"total={bytes_to_gib(total):.2f} GiB "
        f"allocated={allocated:.1f} MiB reserved={reserved:.1f} MiB"
    )


@dataclass
class DatasetInfo:
    name: str
    dataset: torch.utils.data.Dataset


def build_dataset(data_root: str, train_size: int, eval_size: int) -> tuple[DatasetInfo, DatasetInfo]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    try:
        train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test = datasets.MNIST(data_root, train=False, download=True, transform=transform)
        print(f"Using torchvision MNIST at {data_root}")
        return (
            DatasetInfo("mnist-train", Subset(train, range(min(train_size, len(train))))),
            DatasetInfo("mnist-test", Subset(test, range(min(eval_size, len(test))))),
        )
    except Exception as exc:
        print(f"MNIST download/load failed: {exc}")
        print("Falling back to torchvision FakeData so CUDA can still be tested.")
        fake_train = datasets.FakeData(
            size=train_size,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        fake_test = datasets.FakeData(
            size=eval_size,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        return DatasetInfo("fake-train", fake_train), DatasetInfo("fake-test", fake_test)


class SmallConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    seed = int(os.environ.get("MNIST_GPU_TEST_SEED", "42"))
    train_size = int(os.environ.get("MNIST_GPU_TRAIN_SAMPLES", "4096"))
    eval_size = int(os.environ.get("MNIST_GPU_EVAL_SAMPLES", "1024"))
    batch_size = int(os.environ.get("MNIST_GPU_BATCH_SIZE", "128"))
    epochs = int(os.environ.get("MNIST_GPU_EPOCHS", "2"))
    data_root = os.environ.get(
        "MNIST_GPU_DATA_ROOT",
        "/home1/doyoonkim/projects/RAC/open-r1-main/data/mnist",
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=== Environment ===")
    print(f"Hostname: {platform.node()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        print_cuda_status("startup", device)

    train_info, test_info = build_dataset(data_root, train_size=train_size, eval_size=eval_size)
    train_loader = DataLoader(
        train_info.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_info.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = SmallConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("=== Training ===")
    print(f"Train dataset: {train_info.name} ({len(train_info.dataset)} samples)")
    print(f"Eval dataset: {test_info.name} ({len(test_info.dataset)} samples)")
    print(f"Batch size: {batch_size}, epochs: {epochs}")
    print_cuda_status("after model.to", device)

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for step, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_count = labels.size(0)
            running_loss += loss.item() * batch_count
            seen += batch_count

            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )
                print_cuda_status(f"epoch {epoch} step {step}", device)

        avg_loss = running_loss / max(seen, 1)
        acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch} avg_loss={avg_loss:.4f} eval_acc={acc:.4f}")

    elapsed = time.time() - start
    print("=== Finished ===")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print_cuda_status("final", device)
    run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])


if __name__ == "__main__":
    main()
