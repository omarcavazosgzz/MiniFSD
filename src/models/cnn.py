import torch
import torch.nn as nn


class SteeringCNN(nn.Module):
    """
    Baseline CNN inspired by NVIDIA's end-to-end driving model.

    Input:  RGB image tensor [B, 3, 66, 200]
    Output: steering value [B, 1]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(100, 50),
            nn.ELU(),

            nn.Linear(50, 10),
            nn.ELU(),

            nn.Linear(10, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


if __name__ == "__main__":
    model = SteeringCNN()
    dummy = torch.randn(4, 3, 66, 200)
    output = model(dummy)

    print("Input:", dummy.shape)
    print("Output:", output.shape)

