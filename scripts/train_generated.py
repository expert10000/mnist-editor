from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = ROOT / "generated"
DEFAULT_METRICS_PATH = DEFAULT_GENERATED_DIR / "train_metrics.json"
DEFAULT_CHECKPOINT_PATH = DEFAULT_GENERATED_DIR / "generated_mnist.pt"
DATA_DIR = ROOT / "data"


@dataclass
class TrainMetrics:
    status: str
    run_id: str
    topology_id: str
    created_at: str
    updated_at: str
    epochs: int
    train_limit: int
    test_limit: int
    seed: int
    current_epoch: int
    current_batch: int
    total_batches: int
    first_batch_loss: float
    final_batch_loss: float
    final_batch_accuracy: float
    train_loss: float
    train_accuracy: float
    baseline_test_loss: float
    baseline_accuracy: float
    test_loss: float
    test_accuracy: float
    best_accuracy: float
    best_epoch: int
    accuracy_delta: float
    epoch_history: list[dict[str, float | int]]
    checkpoint: str
    duration_seconds: float
    passed_smoke_rule: bool


def load_generated_model(generated_dir: Path) -> nn.Module:
    model_path = generated_dir / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} is missing. Generate model.py first.")

    sys.path.insert(0, str(generated_dir))
    spec = importlib.util.spec_from_file_location("generated_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {model_path}.")
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
    created_at = args.created_at or datetime.now(UTC).isoformat()
    generated_dir = resolve_path(args.generated_dir, DEFAULT_GENERATED_DIR)
    run_dir = resolve_path(args.run_dir, generated_dir)
    metrics_path = resolve_path(args.metrics_path, run_dir / "metrics.json" if args.run_dir else DEFAULT_METRICS_PATH)
    latest_metrics_path = resolve_path(args.latest_metrics_path, DEFAULT_METRICS_PATH)
    checkpoint_path = resolve_path(args.checkpoint_path, run_dir / "checkpoint.pt" if args.run_dir else DEFAULT_CHECKPOINT_PATH)
    latest_checkpoint_path = resolve_path(args.latest_checkpoint_path, DEFAULT_CHECKPOINT_PATH)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    generated_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    latest_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    progress: dict[str, Any] = {
        "status": "running",
        "run_id": args.run_id,
        "topology_id": args.topology_id,
        "created_at": created_at,
        "updated_at": created_at,
        "epochs": args.epochs,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "seed": args.seed,
        "current_epoch": 0,
        "current_batch": 0,
        "total_batches": 0,
        "first_batch_loss": None,
        "final_batch_loss": None,
        "final_batch_accuracy": None,
        "train_loss": None,
        "train_accuracy": None,
        "baseline_test_loss": None,
        "baseline_accuracy": None,
        "test_loss": None,
        "test_accuracy": None,
        "best_accuracy": None,
        "best_epoch": 0,
        "accuracy_delta": None,
        "epoch_history": [],
        "checkpoint": relative_path(checkpoint_path),
        "duration_seconds": 0.0,
        "passed_smoke_rule": None,
    }
    write_progress(progress, metrics_path, latest_metrics_path)
    print(f"run {args.run_id} started on {device}", flush=True)

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
    total_batches = len(train_loader) * args.epochs
    progress["total_batches"] = total_batches
    write_progress(progress, metrics_path, latest_metrics_path)
    print(f"loaded MNIST subset: train={args.train_limit} test={args.test_limit} batches={total_batches}", flush=True)

    model = load_generated_model(generated_dir).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    baseline_test_loss, baseline_accuracy = evaluate(model, test_loader, criterion, device)
    best_accuracy = baseline_accuracy
    best_epoch = 0
    epoch_history: list[dict[str, float | int]] = []
    progress.update(
        {
            "baseline_test_loss": baseline_test_loss,
            "baseline_accuracy": baseline_accuracy,
            "test_loss": baseline_test_loss,
            "test_accuracy": baseline_accuracy,
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "accuracy_delta": 0.0,
            "duration_seconds": round(time.perf_counter() - start, 2),
        }
    )
    write_progress(progress, metrics_path, latest_metrics_path)
    print(f"baseline accuracy={baseline_accuracy:.4f} test_loss={baseline_test_loss:.4f}", flush=True)

    first_batch_loss: float | None = None
    final_batch_loss = 0.0
    final_batch_accuracy = 0.0
    running_loss = 0.0
    running_correct = 0
    seen = 0
    global_batch = 0
    test_loss = baseline_test_loss
    test_accuracy = baseline_accuracy

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_seen = 0
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
            batch_correct = int((logits.argmax(dim=1) == labels).sum().item())
            final_batch_accuracy = batch_correct / max(1, images.size(0))
            final_batch_loss = batch_loss
            running_loss += batch_loss * images.size(0)
            running_correct += batch_correct
            seen += images.size(0)
            epoch_loss += batch_loss * images.size(0)
            epoch_seen += images.size(0)
            global_batch += 1

            progress.update(
                {
                    "current_epoch": epoch,
                    "current_batch": global_batch,
                    "first_batch_loss": first_batch_loss,
                    "final_batch_loss": final_batch_loss,
                    "final_batch_accuracy": final_batch_accuracy,
                    "train_loss": running_loss / max(1, seen),
                    "train_accuracy": running_correct / max(1, seen),
                    "duration_seconds": round(time.perf_counter() - start, 2),
                }
            )
            write_progress(progress, metrics_path, latest_metrics_path)
            print(f"batch {global_batch}/{total_batches} loss={batch_loss:.4f}", flush=True)

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        epoch_train_loss = epoch_loss / max(1, epoch_seen)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": float(epoch_train_loss),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
            }
        )
        progress.update(
            {
                "train_loss": running_loss / max(1, seen),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "accuracy_delta": best_accuracy - baseline_accuracy,
                "epoch_history": epoch_history,
                "duration_seconds": round(time.perf_counter() - start, 2),
            }
        )
        write_progress(progress, metrics_path, latest_metrics_path)
        print(
            f"epoch {epoch}/{args.epochs} accuracy={test_accuracy:.4f} "
            f"best={best_accuracy:.4f} delta={best_accuracy - baseline_accuracy:+.4f} test_loss={test_loss:.4f}",
            flush=True,
        )

    train_loss = running_loss / max(1, seen)
    train_accuracy = running_correct / max(1, seen)
    passed_smoke_rule = bool(best_accuracy >= args.min_accuracy and first_batch_loss is not None and final_batch_loss < first_batch_loss)
    status = "complete" if passed_smoke_rule else "failed"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "accuracy_delta": best_accuracy - baseline_accuracy,
            },
            "args": vars(args),
        },
        checkpoint_path,
    )
    if checkpoint_path != latest_checkpoint_path:
        shutil.copyfile(checkpoint_path, latest_checkpoint_path)

    metrics = TrainMetrics(
        status=status,
        run_id=args.run_id,
        topology_id=args.topology_id,
        created_at=created_at,
        updated_at=datetime.now(UTC).isoformat(),
        epochs=args.epochs,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        current_epoch=args.epochs,
        current_batch=global_batch,
        total_batches=total_batches,
        first_batch_loss=float(first_batch_loss if first_batch_loss is not None else 0.0),
        final_batch_loss=float(final_batch_loss),
        final_batch_accuracy=float(final_batch_accuracy),
        train_loss=float(train_loss),
        train_accuracy=float(train_accuracy),
        baseline_test_loss=float(baseline_test_loss),
        baseline_accuracy=float(baseline_accuracy),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        best_accuracy=float(best_accuracy),
        best_epoch=best_epoch,
        accuracy_delta=float(best_accuracy - baseline_accuracy),
        epoch_history=epoch_history,
        checkpoint=relative_path(checkpoint_path),
        duration_seconds=round(time.perf_counter() - start, 2),
        passed_smoke_rule=passed_smoke_rule,
    )
    write_progress(asdict(metrics), metrics_path, latest_metrics_path)
    print(f"run {args.run_id} {status}: accuracy={test_accuracy:.4f} best={best_accuracy:.4f} train_loss={train_loss:.4f}", flush=True)
    if not metrics.passed_smoke_rule:
        raise RuntimeError(
            f"Generated model training check failed: accuracy={test_accuracy:.3f}, "
            f"first_loss={metrics.first_batch_loss:.3f}, final_loss={final_batch_loss:.3f}"
        )
    return metrics


def write_progress(payload: dict[str, Any], metrics_path: Path, latest_metrics_path: Path) -> None:
    payload = {**payload, "updated_at": datetime.now(UTC).isoformat()}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf8")
    if metrics_path != latest_metrics_path:
        latest_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf8")


def resolve_path(value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
    parser.add_argument("--generated-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--metrics-path")
    parser.add_argument("--latest-metrics-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--latest-checkpoint-path")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--topology-id", default="default")
    parser.add_argument("--created-at")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
