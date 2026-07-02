from pathlib import Path
from datetime import datetime
import csv
import cv2


class DataRecorder:
    def __init__(self, session_name=None, session_prefix="session"):
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"{session_prefix}_{timestamp}"

        self.base_dir = Path("data/raw") / session_name
        self.images_dir = self.base_dir / "images"

        if self.base_dir.exists():
            raise FileExistsError(
                f"Session folder already exists: {self.base_dir}. "
                "Use a new session name to avoid overwriting data."
            )

        self.images_dir.mkdir(parents=True, exist_ok=False)

        self.csv_path = self.base_dir / "driving_log.csv"
        self.frame_id = 0

        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            "frame",
            "image",
            "steering",
            "throttle",
            "brake",
            "speed",
        ])

    def record(self, image, steering, throttle, brake, speed):
        filename = f"{self.frame_id:06d}.png"
        image_path = self.images_dir / filename

        cv2.imwrite(str(image_path), image)

        self.writer.writerow([
            self.frame_id,
            str(image_path),
            steering,
            throttle,
            brake,
            speed,
        ])

        self.csv_file.flush()
        self.frame_id += 1

    def close(self):
        self.csv_file.close()
