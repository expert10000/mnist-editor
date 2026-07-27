from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "generated"
METRICS_PATH = GENERATED_DIR / "train_metrics.json"
CHECKPOINT_PATH = GENERATED_DIR / "generated_mnist.pt"
DATA_DIR = ROOT / "data"


@dataclass
class TrainMetrics:
    status: str
    epochs: int
    train_limit: int
    test_limit: int
    seed: int
    first_batch_loss: float
    final_batch_loss: float
    train_loss: float
    test_loss: float
    test_accuracy: float
    checkpoint: str
    duration_seconds: float
    passed_smoke_rule: bool


def load_generated_model() -> nn.Module:
    model_path = GENERATED_DIR / "model.py"
    if not model_path.exists():
        raise FileNotFoundError("generated/model.py is missing. Run npm run generate first.")

    sys.path.insert(0, str(GENERATED_DIR))
    spec = importlib.util.spec_from_file_location("generated_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load generated/model.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GeneratedMNISTModel()


def limited_subset(dataset: datasets.MNIST, limit: int, seed: int) -> Subset:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[: min(limit, len(indices))])


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def train(args: argparse.Namespace) -> TrainMetrics:
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        limited_subset(train_dataset, args.train_limit, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        limited_subset(test_dataset, args.test_limit, args.seed + 1),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_generated_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    first_batch_loss: float | None = None
    final_batch_loss = 0.0
    running_loss = 0.0
    seen = 0
    model.train()
    for _epoch in range(args.epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary_logits = model(images, return_auxiliary=True)
            loss = criterion(logits, labels)
            if auxiliary_logits is not None:
                loss = loss + args.aux_weight * criterion(auxiliary_logits, labels)
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.item())
            if first_batch_loss is None:
                first_batch_loss = batch_loss
            final_batch_loss = batch_loss
            running_loss += batch_loss * images.size(0)
            seen += images.size(0)

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    train_loss = running_loss / max(1, seen)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            },
            "args": vars(args),
        },
        CHECKPOINT_PATH,
    )

    metrics = TrainMetrics(
        status="complete",
        epochs=args.epochs,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        first_batch_loss=float(first_batch_loss if first_batch_loss is not None else 0.0),
        final_batch_loss=float(final_batch_loss),
        train_loss=float(train_loss),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        checkpoint=str(CHECKPOINT_PATH.relative_to(ROOT)),
        duration_seconds=round(time.perf_counter() - start, 2),
        passed_smoke_rule=bool(test_accuracy >= args.min_accuracy and first_batch_loss is not None and final_batch_loss < first_batch_loss),
    )
    METRICS_PATH.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf8")
    if not metrics.passed_smoke_rule:
        raise RuntimeError(
            f"Generated model training check failed: accuracy={test_accuracy:.3f}, "
            f"first_loss={metrics.first_batch_loss:.3f}, final_loss={final_batch_loss:.3f}"
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the generated MNIST model on a small real MNIST subset.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-accuracy", type=float, default=0.12)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(asdict(result), indent=2))
