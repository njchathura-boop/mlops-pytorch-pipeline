"""Train a CIFAR-10 model and save the best checkpoint."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

from src.dataset import CIFAR10_CLASSES, get_dataloaders
from src.model import get_model


def log_json(values: dict[str, Any]) -> None:
    """Print one structured JSON line."""

    print(json.dumps(values), flush=True)


def resolve_config_path() -> Path:
    """Locate the mounted, environment-provided, or local config file."""

    configured = os.getenv("TRAINING_CONFIG_PATH")
    candidates = [
        Path(configured) if configured else None,
        Path("/app/configs/training_config.yaml"),
        Path("configs/training_config.yaml"),
    ]

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "training_config.yaml was not found. Set TRAINING_CONFIG_PATH."
    )


def load_config(path: Path) -> dict[str, Any]:
    """Read the YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required = {"model", "training", "data", "output"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    return config


def set_seed(seed: int) -> None:
    """Set basic random seeds for repeatability."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch."""

    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    architecture: str,
    num_classes: int,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
) -> None:
    """Save model weights and serving metadata."""

    torch.save(
        {
            "architecture": architecture,
            "num_classes": num_classes,
            "class_names": CIFAR10_CLASSES,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def main() -> None:
    config_path = resolve_config_path()
    config = load_config(config_path)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]

    set_seed(int(training_config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architecture = str(model_config["architecture"])
    num_classes = int(model_config["num_classes"])
    model = get_model(architecture, num_classes).to(device)

    train_loader, validation_loader = get_dataloaders(
        data_dir=str(data_config["data_dir"]),
        batch_size=int(training_config["batch_size"]),
        num_workers=int(training_config.get("num_workers", 0)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / str(output_config["model_name"])

    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    min_delta = float(training_config.get("min_delta", 0.0))

    best_validation_loss = float("inf")
    patience_counter = 0

    log_json(
        {
            "event": "training_started",
            "device": str(device),
            "architecture": architecture,
            "config": str(config_path),
        }
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        log_json(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_accuracy, 4),
                "validation_loss": round(validation_loss, 4),
                "validation_accuracy": round(validation_accuracy, 4),
            }
        )

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            patience_counter = 0
            save_checkpoint(
                checkpoint_path,
                model,
                architecture,
                num_classes,
                epoch,
                validation_loss,
                validation_accuracy,
            )
            log_json(
                {
                    "event": "checkpoint_saved",
                    "path": str(checkpoint_path),
                    "epoch": epoch,
                }
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_json({"event": "early_stopping", "epoch": epoch})
                break

    log_json(
        {
            "event": "training_complete",
            "best_validation_loss": round(best_validation_loss, 4),
            "checkpoint": str(checkpoint_path),
        }
    )


if __name__ == "__main__":
    main()
