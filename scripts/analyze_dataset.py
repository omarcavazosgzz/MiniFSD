import argparse
from pathlib import Path

import pandas as pd


def find_latest_session(base_dir: Path) -> Path:
    sessions = [p for p in base_dir.iterdir() if p.is_dir()]

    if not sessions:
        raise FileNotFoundError(f"No sessions found in {base_dir}")

    return max(sessions, key=lambda p: p.stat().st_mtime)


def resolve_image_path(session_dir: Path, image_value: str) -> Path:
    image_path = Path(image_value)

    if image_path.exists():
        return image_path

    fallback_path = session_dir / "images" / image_path.name

    if fallback_path.exists():
        return fallback_path

    return image_path


def analyze_session(session_dir: Path):
    csv_path = session_dir / "driving_log.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing driving_log.csv in {session_dir}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "frame",
        "image",
        "steering",
        "throttle",
        "brake",
        "speed",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    image_paths = [
        resolve_image_path(session_dir, image_value)
        for image_value in df["image"]
    ]

    missing_images = [path for path in image_paths if not path.exists()]

    rows = len(df)
    throttle_frames = int((df["throttle"] > 0.05).sum())
    brake_frames = int((df["brake"] > 0.05).sum())
    steering_frames = int((df["steering"].abs() > 0.01).sum())

    left_frames = int((df["steering"] < -0.02).sum())
    right_frames = int((df["steering"] > 0.02).sum())
    straight_frames = int((df["steering"].abs() <= 0.02).sum())

    moving_frames = int((df["speed"] > 1.0).sum())
    stopped_frames = int((df["speed"] <= 1.0).sum())

    print()
    print("MiniFSD Dataset Analysis")
    print("=" * 40)
    print(f"Session: {session_dir}")
    print(f"Rows: {rows}")
    print(f"Missing images: {len(missing_images)}")
    print()

    print("Control Activity")
    print("-" * 40)
    print(f"Throttle frames: {throttle_frames} ({throttle_frames / rows * 100:.1f}%)")
    print(f"Brake frames:    {brake_frames} ({brake_frames / rows * 100:.1f}%)")
    print(f"Steering frames: {steering_frames} ({steering_frames / rows * 100:.1f}%)")
    print()

    print("Steering Balance")
    print("-" * 40)
    print(f"Left frames:     {left_frames} ({left_frames / rows * 100:.1f}%)")
    print(f"Right frames:    {right_frames} ({right_frames / rows * 100:.1f}%)")
    print(f"Straight frames: {straight_frames} ({straight_frames / rows * 100:.1f}%)")
    print()

    print("Speed")
    print("-" * 40)
    print(f"Moving frames:   {moving_frames} ({moving_frames / rows * 100:.1f}%)")
    print(f"Stopped frames:  {stopped_frames} ({stopped_frames / rows * 100:.1f}%)")
    print(f"Mean speed:      {df['speed'].mean():.2f} km/h")
    print(f"Max speed:       {df['speed'].max():.2f} km/h")
    print()

    print("Steering Summary")
    print("-" * 40)
    print(df["steering"].describe())
    print()

    print("Recommendation")
    print("-" * 40)

    if rows < 1000:
        print("- Dataset is small. Good for testing, not enough for training.")

    if missing_images:
        print("- Fix missing images before training.")

    if steering_frames < rows * 0.20:
        print("- Add more turns. Too much straight driving.")

    if left_frames == 0 or right_frames == 0:
        print("- Dataset needs both left and right turns.")
    else:
        balance_ratio = min(left_frames, right_frames) / max(left_frames, right_frames)

        if balance_ratio < 0.40:
            print("- Steering is unbalanced. Add more turns in the weaker direction.")
        else:
            print("- Steering balance is acceptable.")

    if throttle_frames < rows * 0.40:
        print("- Add more moving frames with throttle.")

    if brake_frames < rows * 0.02:
        print("- Add some braking examples.")

    if not missing_images and rows >= 1000:
        print("- Dataset is usable for early training experiments.")

    print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Path to a specific dataset session folder."
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Analyze the most recently modified session in data/raw."
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

    analyze_session(session_dir)


if __name__ == "__main__":
    main()
