import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

from src.dataset.carla_dataset import CarlaDrivingDataset
from src.models.cnn import SteeringCNN


def find_latest_session(base_dir: Path) -> Path:
    sessions = [p for p in base_dir.iterdir() if p.is_dir()]

    if not sessions:
        raise FileNotFoundError(f"No sessions found in {base_dir}")

    return max(sessions, key=lambda p: p.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session",
        type=str,
        action="append",
        default=None,
        help="Dataset session folder. Can be used multiple times.",
    )

    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest session in data/raw.",
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--checkpoint-name", type=str, default="steering_cnn_best.pt")

    return parser.parse_args()


def build_dataset(args):
    if args.latest:
        session_dirs = [find_latest_session(Path("data/raw"))]
    elif args.session:
        session_dirs = [Path(session) for session in args.session]
    else:
        raise ValueError("Use --latest or one/more --session path/to/session")

    datasets = []

    for session_dir in session_dirs:
        if not session_dir.exists():
            raise FileNotFoundError(f"Session folder not found: {session_dir}")

        dataset = CarlaDrivingDataset(session_dir)
        datasets.append(dataset)

    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = ConcatDataset(datasets)

    return combined_dataset, session_dirs


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for images, controls in loader:
        images = images.to(device)
        steering_targets = controls[:, 0:1].to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_fn(predictions, steering_targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_mae += torch.abs(predictions - steering_targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_mae / total_samples


def validate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, controls in loader:
            images = images.to(device)
            steering_targets = controls[:, 0:1].to(device)

            predictions = model(images)
            loss = loss_fn(predictions, steering_targets)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_mae += torch.abs(predictions - steering_targets).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_mae / total_samples


def main():
    args = parse_args()

    dataset, session_dirs = build_dataset(args)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SteeringCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    best_val_loss = float("inf")

    print("MiniFSD Training")
    print("=" * 40)
    print("Sessions:")
    for session_dir in session_dirs:
        print(f"  - {session_dir}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        val_loss, val_mae = validate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"train_mae={train_mae:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_mae={val_mae:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "session_dirs": [str(session_dir) for session_dir in session_dirs],
                    "image_size": [66, 200],
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint: {checkpoint_path}")

    print()
    print("Training finished.")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()