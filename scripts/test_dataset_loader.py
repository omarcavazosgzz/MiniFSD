import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset.carla_dataset import CarlaDrivingDataset


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
        default=None,
        help="Path to dataset session folder.",
    )

    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest session in data/raw.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.latest:
        session_dir = find_latest_session(Path("data/raw"))
    elif args.session:
        session_dir = Path(args.session)
    else:
        raise ValueError("Use --latest or --session path/to/session")

    dataset = CarlaDrivingDataset(session_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    images, controls = next(iter(dataloader))

    print("Dataset loaded correctly.")
    print(f"Session: {session_dir}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Control batch shape: {controls.shape}")
    print()
    print("First control sample:")
    print(f"Steering: {controls[0][0].item():.4f}")
    print(f"Throttle: {controls[0][1].item():.4f}")
    print(f"Brake:    {controls[0][2].item():.4f}")


if __name__ == "__main__":
    main()
