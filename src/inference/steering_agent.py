from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.cnn import SteeringCNN


class SteeringAgent:
    def __init__(
        self,
        checkpoint_path="models/checkpoints/steering_cnn_best.pt",
        device=None,
        image_width=200,
        image_height=66,
        crop_top_ratio=0.35,
        crop_bottom_ratio=0.05,
        max_steer=0.7,
    ):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.image_width = image_width
        self.image_height = image_height
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.max_steer = max_steer

        self.model = SteeringCNN().to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Loaded model from: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Checkpoint val_loss: {checkpoint.get('val_loss')}")
        print(f"Checkpoint val_mae: {checkpoint.get('val_mae')}")

    def preprocess(self, image):
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
        image = image.unsqueeze(0)

        return image.to(self.device)

    @torch.no_grad()
    def predict(self, image):
        tensor = self.preprocess(image)
        steering = self.model(tensor).item()

        steering = float(np.clip(steering, -self.max_steer, self.max_steer))

        return steering
