from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class CarlaDrivingDataset(Dataset):
    def __init__(
        self,
        session_dir,
        image_width=200,
        image_height=66,
        crop_top_ratio=0.35,
        crop_bottom_ratio=0.05,
    ):
        self.session_dir = Path(session_dir)
        self.csv_path = self.session_dir / "driving_log.csv"

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        self.image_width = image_width
        self.image_height = image_height
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio

        required_columns = ["image", "steering", "throttle", "brake"]

        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(f"Missing required column: {column}")

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, image_value):
        image_path = Path(image_value)

        if image_path.exists():
            return image_path

        fallback_path = self.session_dir / "images" / image_path.name

        if fallback_path.exists():
            return fallback_path

        raise FileNotFoundError(f"Image not found: {image_value}")

    def _preprocess_image(self, image):
        height = image.shape[0]

        top = int(height * self.crop_top_ratio)
        bottom = int(height * (1.0 - self.crop_bottom_ratio))

        image = image[top:bottom, :]

        image = cv2.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_AREA,
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype("float32") / 255.0

        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self._resolve_image_path(row["image"])
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        image = self._preprocess_image(image)

        control = torch.tensor(
            [
                float(row["steering"]),
                float(row["throttle"]),
                float(row["brake"]),
            ],
            dtype=torch.float32,
        )

        return image, control
