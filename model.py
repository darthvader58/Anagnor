import torch
import torch.nn as nn
import torch.nn.functional as F


class AnagnorModel(nn.Module):
    """Binary landslide-probability classifier over a 32-channel raster stack."""

    def __init__(self, in_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 12, 3)
        self.pool = nn.MaxPool2d(10, 10)
        self.conv2 = nn.Conv2d(12, 3, 3)
        self.fc1 = nn.Linear(300, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Return logits; pair with BCEWithLogitsLoss during training and apply
        # torch.sigmoid at inference time for probabilities.
        return self.fc3(x)


if __name__ == "__main__":
    net = AnagnorModel()
    p = net(torch.randn(4, 32, 1024, 1024))
    print(p.shape, torch.sigmoid(p))
